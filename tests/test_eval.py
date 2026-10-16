"""Phase 5: evaluation pipeline.

The key tests compare the independent CPU MuJoCo runner (built only from the
exported policy file) with the training environment:
  * identical observation vector for the same physical state
  * the same closed-loop trajectory for 0.5 s (MJX float32 vs MuJoCo float64)
plus an end-to-end run of the MJX suite and the sim-to-sim runner.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import RUNTIME
from g1_rl import checkpoint, evaluate, ppo
from g1_rl.config import EnvConfig, NoiseConfig, PPOConfig, SimConfig
from g1_rl.env import G1Env
from g1_rl.policy import Policy
from g1_rl.sim2sim import MujocoRunner

CFG = PPOConfig(policy_hidden=(64, 64), value_hidden=(64, 64))


def make_policy(directory, gain):
  """An untrained policy whose output layer is scaled by `gain` (larger = bigger actions)."""
  env = G1Env()
  ts = ppo.init_train_state(jax.random.PRNGKey(0), env.obs_size, env.priv_obs_size,
                            env.action_size, CFG)
  net = ts.params["policy"]["net"]
  net[-1] = {"w": net[-1]["w"] * gain, "b": net[-1]["b"]}
  path = str(directory / f"policy_{gain}.npz")
  checkpoint.export_policy(path, ts, env, CFG)
  return Policy.load(path, device=jax.devices()[0])


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
  return make_policy(tmp_path_factory.mktemp("pol"), 40)  # saturating, ~0.6 mean |action|


def _deterministic_env():
  cfg = evaluate.eval_env_config(EnvConfig(), 1000, obs_noise=False)
  return G1Env(dataclasses.replace(cfg, dr=dataclasses.replace(cfg.dr, level=0.0)))


def _fixed_episode(env, state, command=(0.4, 0.0, 0.2)):
  info = dict(state.info)
  info["command"] = jnp.asarray(command, jnp.float32)
  info["cmd_timer"] = jnp.asarray(10 ** 7, jnp.int32)
  info["push_timer"] = jnp.asarray(10 ** 7, jnp.int32)
  return jax.jit(env.refresh_obs)(state._replace(info=info), jax.random.PRNGKey(9))


def _copy_into_runner(runner, env, state):
  d = env.packer.unpack(state.data)
  runner.d.qpos[:] = np.asarray(d.qpos, np.float64)
  runner.d.qvel[:] = np.asarray(d.qvel, np.float64)
  runner.d.ctrl[:] = np.asarray(d.ctrl, np.float64)
  import mujoco
  mujoco.mj_forward(runner.m, runner.d)
  runner.command = np.asarray(state.info["command"], np.float64)
  runner.last_action = np.asarray(state.info["last_act"], np.float64)
  runner.phase = np.asarray(state.info["phase"], np.float64)
  runner.gait_freq = float(state.info["phase_dt"]) / (2 * math.pi * env.dt)


def test_observation_matches_training_env(policy):
  env = _deterministic_env()
  step = jax.jit(env.step)
  s = _fixed_episode(env, jax.jit(env.reset)(jax.random.PRNGKey(1)))
  pi = policy.jax_fn()
  for _ in range(5):  # move a little so velocities and last action are non-trivial
    s = step(s, pi(s.obs))
  runner = MujocoRunner(SimConfig().xml_path, policy)
  _copy_into_runner(runner, env, s)
  np.testing.assert_allclose(runner.observation(), np.asarray(s.obs), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("gain,steps", [(40, 8), (10, 25)])
def test_closed_loop_matches_mjx(tmp_path, gain, steps):
  """Same start, same policy: MJX (float32) and CPU MuJoCo (float64) stay together.

  With a saturating random policy (gain 40) the two agree to ~1e-3 for 8 steps and
  then diverge chaotically (bang-bang control at contact events amplifies rounding),
  so only the first 8 steps are compared; a moderate policy is compared for 0.5 s.
  """
  policy = make_policy(tmp_path, gain)
  env = _deterministic_env()
  step = jax.jit(env.step)
  s = _fixed_episode(env, jax.jit(env.reset)(jax.random.PRNGKey(2)))
  runner = MujocoRunner(SimConfig().xml_path, policy)
  _copy_into_runner(runner, env, s)
  pi = policy.jax_fn()
  for t in range(steps):
    a_mjx = np.asarray(pi(s.obs))
    a_cpu = runner.step()
    np.testing.assert_allclose(a_cpu, a_mjx, atol=0.01, err_msg=f"step {t}")
    s = step(s, jnp.asarray(a_mjx))
    assert float(s.done) == 0
  q_mjx = np.asarray(env.packer.unpack(s.data).qpos)
  q_cpu = runner.d.qpos
  assert np.abs(q_mjx[:3] - q_cpu[:3]).max() < 0.005  # base position (m)
  assert np.abs(q_mjx[7:] - q_cpu[7:]).max() < 0.005  # joint angles (rad)


def test_full_model_runs(policy):
  runner = MujocoRunner(SimConfig().eval_xml_path, policy)
  assert runner.n_sub == 10 and runner.m.npair == 0  # 2 ms, mesh contacts
  np.testing.assert_allclose(runner.m.actuator_gainprm[runner.act, 0], policy.kp)  # policy gains used
  runner.reset((0.5, 0, 0))
  assert abs(runner._lowest_foot_point()) < 1e-6  # placed exactly on the floor
  out = runner.run((0.5, 0, 0), seconds=1.0)
  assert out["time_alive_s"] > 0.2


def test_mjx_suite_end_to_end(policy):
  sc = [evaluate.Scenario("grid", "fwd", (0.5, 0, 0), tag={"vx": 0.5, "vy": 0.0, "wz": 0.0}),
        evaluate.Scenario("training", "train", dr_level=1.0, resample_commands=True, random_pushes=True),
        evaluate.Scenario("ood", "friction=0.3", (0.5, 0, 0), dr_override={"friction": 0.3},
                          tag={"param": "friction", "value": 0.3}),
        evaluate.Scenario("push", "stand/front/3.0", (0, 0, 0), push_time=0.2, push_vel=(3.0, 0.0),
                          tag={"cmd": "stand", "dir": "front", "mag": 3.0})]
  metrics, extra = evaluate.run_mjx_suite(policy, RUNTIME, sc, seconds=2.0, record_qpos_for=(0,))
  assert [m["group"] for m in metrics] == ["grid", "training", "ood", "push"]
  push, grid = metrics[3], metrics[0]
  assert not push["survived"] and push["time_alive_s"] < grid["time_alive_s"]  # a 3 m/s shove topples it
  assert extra["qpos"][0].shape == (100, 36)
  summary = evaluate.summarise(metrics)
  assert set(summary) == {"grid", "training", "ood", "push"}
  assert summary["push"]["stand/front"]["max_recovered_mps"] == 0.0
