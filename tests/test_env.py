"""Phase 1 tests for G1Env. Uses one batched reset and one batched step compile.

Each env in a batch of 4 is put in a different situation before a single step:
  env 0: body rotated onto its side                  -> fall
  env 1: untouched                                   -> normal step
  env 2: step counter at the time limit - 1          -> timeout (truncation)
  env 3: command timer about to expire               -> command resampled
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import RUNTIME
from g1_rl import parallel
from g1_rl.config import EnvConfig
from g1_rl.env import G1Env
from g1_rl.vec_env import VecEnv

N = 4 if RUNTIME.n_devices <= 4 else RUNTIME.n_devices


@pytest.fixture(scope="module")
def setup():
  env = G1Env(EnvConfig())
  venv = VecEnv(env, RUNTIME.mesh, N)
  s0 = venv.reset(seed=0)
  return env, venv, s0


def test_layout(setup):
  env, _, _ = setup
  assert env.obs_size == 100 and env.priv_obs_size == 139
  assert env.priv_layout["obs"] == slice(0, 100)


def test_reset(setup):
  env, venv, s0 = setup
  s = jax.device_get(s0)
  assert s.obs.shape == (N, 100) and s.priv_obs.shape == (N, 139)
  assert np.isfinite(s.obs).all() and np.isfinite(s.priv_obs).all()
  g = s.obs[:, env.obs_layout["gravity"]]
  assert (g[:, 2] < -0.9).all()  # upright (+ obs noise)
  fh = s.priv_obs[:, env.priv_layout["foot_height"]]
  assert (np.abs(fh.min(1)) < 5e-3).all() and (fh.max(1) < 0.03).all()
  c = env.cfg.command
  cmd = s.info["command"]
  assert (cmd[:, 0] >= c.lin_vel_x[0]).all() and (cmd[:, 0] <= c.lin_vel_x[1]).all()
  assert (s.info["step"] == 0).all() and (s.done == 0).all()
  # Same seed -> identical reset.
  s_again = jax.device_get(venv.reset(seed=0))
  np.testing.assert_array_equal(s.obs, s_again.obs)


def test_step_cases(setup):
  env, venv, s0 = setup
  # VecEnv.step donates its input state (frees the buffers for reuse on GPU).
  # Snapshot everything we compare against to host memory first, and give step
  # its own copy so the shared fixture `s0` is not invalidated for other tests.
  s0_host = jax.device_get(s0)
  s0 = jax.tree.map(jnp.copy, s0)
  pk = env.packer
  d = jax.vmap(pk.unpack)(s0.data)
  # env 0: rotate the body 90 deg about x (lying on its side) -> tilt termination.
  qpos = d.qpos.at[0, 3:7].set(jnp.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]))
  d = d.replace(qpos=qpos)
  info = dict(s0.info)
  info["step"] = info["step"].at[2].set(env.cfg.episode_length - 1)
  info["cmd_timer"] = info["cmd_timer"].at[3].set(1)
  old_cmd = np.asarray(info["command"])
  s = s0._replace(data=jax.vmap(pk.pack)(d), info=info)
  s = parallel.put_batched(s, RUNTIME.mesh)
  out = jax.device_get(venv.step(s, parallel.put_batched(jnp.zeros((N, env.action_size)), RUNTIME.mesh)))

  assert out.done[0] == 1 and out.truncation[0] == 0  # fell
  assert out.info["step"][0] == 0  # auto-reset happened
  assert out.done[1] == 0 and out.info["step"][1] == 1  # normal
  assert out.done[2] == 1 and out.truncation[2] == 1  # timeout
  assert out.metrics["episode/fell"][0] == 1 and out.metrics["episode/fell"][2] == 0
  assert out.metrics["episode/length"][2] == env.cfg.episode_length
  assert not np.allclose(out.info["command"][3], old_cmd[3])  # resampled
  assert out.info["cmd_timer"][3] >= env.cfg.command.resample_steps[0]
  np.testing.assert_allclose(out.info["command"][1], old_cmd[1])  # unchanged
  # Gait phase advanced by phase_dt on the normal env.
  dphi = (out.info["phase"][1] - s0_host.info["phase"][1] + np.pi) % (2 * np.pi) - np.pi
  np.testing.assert_allclose(dphi, s0_host.info["phase_dt"][1], atol=1e-5)
  assert np.isfinite(out.obs).all() and np.isfinite(out.reward).all()
