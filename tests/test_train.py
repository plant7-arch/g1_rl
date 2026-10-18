"""Phase 4 integration: the real training step on the real env, tiny sizes.

Covers: compile + run, metrics finite, iteration counting, curriculum level
handling, checkpoint at the end, resume continues from the saved iteration, and
the exported policy reproduces the trained network. ~2-3 min (one compile;
the resume reuses the persistent compile cache).
"""

import json
import os

import jax
import numpy as np
import pytest

from conftest import RUNTIME
from g1_rl.config import PPOConfig
from g1_rl.policy import Policy
from g1_rl.trainer import train

NDEV = RUNTIME.n_devices
CFG = PPOConfig(num_envs=4 * NDEV, unroll_length=4, num_minibatches=2, num_epochs=2,
                total_env_steps=4 * NDEV * 4 * 3, policy_hidden=(32,), value_hidden=(32,),
                log_every=1, ckpt_every=100, dr_level_start=0.3)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
  base = tmp_path_factory.mktemp("runs")
  a = train(CFG, str(base / "a"), RUNTIME, stop_after=2, video=False)
  b = train(CFG, str(base / "b"), RUNTIME, resume=str(base / "a"), video=False)
  return base, a, b


def test_first_session(runs):
  base, a, _ = runs
  assert a["stopped"] == "stop_after" and a["iterations_done"] == 2
  rows = [json.loads(l) for l in open(base / "a" / "metrics.jsonl")]
  assert [r["iteration"] for r in rows] == [1, 2]
  for k in ("loss/policy", "loss/value", "train/kl", "train/lr", "rollout/reward"):
    assert all(r[k] is not None and np.isfinite(r[k]) for r in rows), k
  assert rows[0]["curriculum/dr_level"] == pytest.approx(0.3)
  assert os.path.exists(base / "a" / "ckpt" / "ckpt_000002.npz")
  assert os.path.exists(base / "a" / "policy_latest.npz")


def test_resume_continues(runs):
  base, _, b = runs
  assert b["stopped"] == "finished" and b["iterations_done"] == 3
  rows = [json.loads(l) for l in open(base / "b" / "metrics.jsonl")]
  assert [r["iteration"] for r in rows] == [3]
  assert os.path.exists(base / "b" / "policy_final.npz")


def test_exported_policy_runs(runs):
  base, _, _ = runs
  pi = Policy.load(str(base / "b" / "policy_final.npz"), device=jax.devices()[0])
  obs = np.zeros((3, pi.meta["obs_size"]), np.float32)
  a = np.asarray(pi(obs))
  assert a.shape == (3, 29) and np.isfinite(a).all() and (np.abs(a) <= 1).all()
  assert pi.meta["iteration"] == 3


@pytest.mark.skipif(NDEV < 2, reason="needs >= 2 devices (on CPU: XLA_FLAGS=--xla_force_host_platform_device_count=2)")
def test_train_step_moves_no_bulk_data_between_devices():
  """Only all-reduces (gradients, small statistics) may cross devices: an all-gather or
  all-to-all would mean rollout/minibatch data is being copied between GPUs."""
  import jax.numpy as jnp
  from jax.sharding import NamedSharding, PartitionSpec as P

  from g1_rl import ppo
  from g1_rl.env import G1Env
  from g1_rl.parallel import count_collectives_text

  env = G1Env()
  N = CFG.num_envs
  ts = jax.eval_shape(lambda: ppo.init_train_state(jax.random.PRNGKey(0), env.obs_size,
                                                   env.priv_obs_size, env.action_size, CFG))
  st = jax.eval_shape(lambda k: jax.vmap(env.reset)(k, jnp.zeros(N)),
                      jax.ShapeDtypeStruct((N, 2), jnp.uint32))
  rep, bat = NamedSharding(RUNTIME.mesh, P()), NamedSharding(RUNTIME.mesh, P("batch"))
  ts = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep), ts)
  st = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=bat), st)
  hlo = ppo.make_train_step(env, CFG, RUNTIME.mesh).lower(ts, st).compile().as_text()
  ops = count_collectives_text(hlo)
  assert set(ops) <= {"all-reduce"}, ops


def test_warm_start_and_critic_warmup(runs, tmp_path):
  """v1.0: a run can start from an exported policy. The policy network and observation
  normaliser are loaded exactly, the learning rate starts at warm_start_lr, and during
  the critic warm-up the policy does not change."""
  import dataclasses
  from g1_rl import checkpoint, ppo
  from g1_rl.env import G1Env
  from g1_rl.trainer import warm_start
  base, _, _ = runs
  src = str(base / "b" / "policy_final.npz")
  cfg = dataclasses.replace(CFG, critic_warmup_iters=2, warm_start_std=0.3, warm_start_lr=2e-4)
  env = G1Env()
  tmpl = ppo.init_train_state(jax.random.PRNGKey(cfg.seed), env.obs_size, env.priv_obs_size, env.action_size, cfg)
  ts = warm_start(tmpl, src, cfg)
  with np.load(src) as z:
    np.testing.assert_allclose(np.asarray(ts.params["policy"]["net"][0]["w"]), z["layer0.w"])
    np.testing.assert_allclose(np.asarray(ts.obs_stats.mean), z["obs_mean"])
  np.testing.assert_allclose(np.exp(np.asarray(ts.params["policy"]["log_std"])), 0.3, rtol=1e-6)
  assert float(ts.lr) == pytest.approx(2e-4)
  # one warm-up iteration: policy frozen, critic trained
  out = train(dataclasses.replace(cfg, total_env_steps=CFG.num_envs * CFG.unroll_length * 1),
              str(tmp_path / "w"), RUNTIME, video=False, init_policy=src)
  assert out["iterations_done"] == 1
  ck = checkpoint.find_checkpoint(str(tmp_path / "w"))
  after, _, _ = checkpoint.load_checkpoint(ck, tmpl)
  np.testing.assert_allclose(np.asarray(after.params["policy"]["net"][0]["w"]),
                             np.asarray(ts.params["policy"]["net"][0]["w"]), atol=1e-7)
  assert not np.allclose(np.asarray(after.params["value"]["net"][0]["w"]),
                         np.asarray(ts.params["value"]["net"][0]["w"]))
  assert float(after.lr) == pytest.approx(2e-4)  # frozen policy: KL 0 must not raise the lr
