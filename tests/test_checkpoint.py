"""Phase 4: checkpoint round trip, pruning, and the standalone policy loader."""

import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g1_rl import checkpoint, networks as nets, normalizer as norm, ppo
from g1_rl.config import PPOConfig
from g1_rl.policy import Policy

CFG = PPOConfig(policy_hidden=(32, 16), value_hidden=(32, 16))


class FakeEnv:
  """Just the attributes export_policy reads (no simulator needed)."""

  def __init__(self):
    from g1_rl import robot
    from g1_rl.config import EnvConfig
    m = robot.load_mj_model()
    self.robot = robot.robot_info(m)
    self.cfg = EnvConfig()
    self.obs_size, self.priv_obs_size, self.action_size, self.dt = 100, 205, 29, 0.02
    self.obs_layout = {"gyro": slice(0, 3), "_size": 100}
    self.frame_size, self.history = 100, 1
    from g1_rl.env import action_scales
    self.action_scale = action_scales(self.robot.actuator_names, self.cfg)

    class DR:
      kp0 = m.actuator_gainprm[:, 0].astype(np.float32)
      kd0 = (-m.actuator_biasprm[:, 2]).astype(np.float32)
    self.dr = DR


def make_state(seed=0):
  ts = ppo.init_train_state(jax.random.PRNGKey(seed), 100, 205, 29, CFG)
  stats = norm.update(ts.obs_stats, jax.random.normal(jax.random.PRNGKey(9), (64, 100)) * 3 + 1)
  return ts._replace(obs_stats=stats, iteration=jnp.asarray(7, jnp.int32), dr_level=jnp.asarray(0.4))


def test_round_trip_and_pruning(tmp_path):
  ts = make_state()
  run = str(tmp_path / "run")
  for it in (5, 10, 15, 20):
    checkpoint.save_checkpoint(run, ts._replace(iteration=jnp.asarray(it, jnp.int32)), {"x": 1}, keep=3)
  files = sorted(os.listdir(os.path.join(run, "ckpt")))
  assert [f for f in files if f.endswith(".npz")] == ["ckpt_000010.npz", "ckpt_000015.npz", "ckpt_000020.npz"]
  template = make_state(seed=1)  # different values, same structure
  loaded, meta, path = checkpoint.load_checkpoint(run, template)
  assert path.endswith("ckpt_000020.npz") and meta["iteration"] == 20 and meta["x"] == 1
  for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(ts._replace(iteration=jnp.asarray(20)))):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
  assert not any(f.endswith(".tmp.npz") for f in files)


def test_shape_mismatch_is_reported(tmp_path):
  checkpoint.save_checkpoint(str(tmp_path), make_state(), {})
  other = ppo.init_train_state(jax.random.PRNGKey(0), 100, 205, 29,
                               dataclasses.replace(CFG, policy_hidden=(64, 16)))
  with pytest.raises(ValueError, match="shape"):
    checkpoint.load_checkpoint(str(tmp_path), other)


def test_find_checkpoint_variants(tmp_path):
  assert checkpoint.find_checkpoint(str(tmp_path)) is None
  p = checkpoint.save_checkpoint(str(tmp_path), make_state(), {})
  assert checkpoint.find_checkpoint(str(tmp_path)) == p
  assert checkpoint.find_checkpoint(os.path.join(str(tmp_path), "ckpt")) == p
  assert checkpoint.find_checkpoint(p) == p


def test_exported_policy_matches_training_network(tmp_path):
  ts = make_state()
  path = checkpoint.export_policy(str(tmp_path / "policy.npz"), ts, FakeEnv(), CFG)
  meta = json.load(open(str(tmp_path / "policy.json")))
  assert meta["format"] == "g1_rl.policy.v1" and len(meta["joint_names"]) == 29
  pi = Policy.load(path, device=jax.devices("cpu")[0])
  obs = jax.random.normal(jax.random.PRNGKey(3), (16, 100)) * 2
  mean, _ = nets.policy_dist(ts.params["policy"], norm.normalize(ts.obs_stats, obs, CFG.obs_clip))
  np.testing.assert_allclose(np.asarray(pi(obs)), np.clip(np.asarray(mean), -1, 1), rtol=1e-5, atol=1e-6)
  t = pi.joint_targets(np.asarray(obs[0]))
  assert t.shape == (29,) and (t >= pi.lo - 1e-6).all() and (t <= pi.hi + 1e-6).all()
