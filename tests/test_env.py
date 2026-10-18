"""Phase 1 tests for G1Env. Uses one batched reset and one batched step compile.

Each env in a batch of 4 is put in a different situation before a single step:
  env 0: body rotated onto its side                  -> fall
  env 1: untouched                                   -> normal step
  env 2: step counter at the time limit - 1          -> timeout (truncation)
  env 3: command timer about to expire               -> command resampled
Randomisation cases folded into the same step:
  env 0: dr_level set to 0 before it falls            -> new episode is the nominal robot
  env 1: latency on, previous action set              -> motors get the previous action; pushed
  env 3: latency off                                  -> motors get the current action
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
  assert env.frame_size == 100 and env.history == 5 and env.obs_size == 500
  assert env.priv_obs_size == 212
  assert env.priv_layout["obs"] == slice(0, 100)
  names = list(env.robot.actuator_names)
  assert env.action_scale[names.index("left_ankle_roll_joint")] == pytest.approx(0.5)
  assert env.action_scale[names.index("left_elbow_joint")] == pytest.approx(0.25)
  assert env.action_scale[names.index("left_knee_joint")] == pytest.approx(0.5)
  body = {env.mj_model.body(i).name: i for i in range(env.mj_model.nbody)}
  ig = env._ignore_pair
  assert ig[body["left_wrist_yaw_link"], body["left_hip_roll_link"]]  # hand-hip: not penalised
  assert not ig[body["left_knee_link"], body["right_knee_link"]]  # legs: penalised
  assert not ig[body["torso_link"], body["left_wrist_yaw_link"]]


def test_reset(setup):
  env, venv, s0 = setup
  s = jax.device_get(s0)
  assert s.obs.shape == (N, env.obs_size) and s.priv_obs.shape == (N, env.priv_obs_size)
  assert np.isfinite(s.obs).all() and np.isfinite(s.priv_obs).all()
  g = s.obs[:, env.obs_layout["gravity"]]
  assert (g[:, 2] < -0.9).all()  # upright (+ obs noise)
  fh = s.priv_obs[:, env.priv_layout["foot_height"]]
  assert (np.abs(fh.min(1)) < 5e-3).all() and (fh.max(1) < 0.03).all()
  c = env.cfg.command
  cmd = s.info["command"]
  assert (cmd[:, 0] >= c.lin_vel_x[0]).all() and (cmd[:, 0] <= c.lin_vel_x[1]).all()
  assert (s.info["step"] == 0).all() and (s.done == 0).all()
  # Default level 1: every env samples its own robot.
  assert len(np.unique(s.info["dr"]["friction"])) == N
  assert (s.info["dr_level"] == env.cfg.dr.level).all()
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
  # randomisation cases
  info["dr_level"] = info["dr_level"].at[0].set(0.0)
  dr = dict(info["dr"])
  dr["latency"] = dr["latency"].at[1].set(1).at[3].set(0)
  info["dr"] = dr
  prev_act = jnp.full(env.action_size, 0.4)
  info["last_act"] = info["last_act"].at[1].set(prev_act)
  info["push_timer"] = info["push_timer"].at[1].set(1).at[2].set(1000).at[3].set(1000)
  kick1 = np.asarray(info["push_kick"][1])
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

  # --- randomisation --------------------------------------------------------
  home, scale = env.robot.home_ctrl, env.action_scale
  lo, hi = env.robot.ctrl_lo, env.robot.ctrl_hi
  np.testing.assert_allclose(out.info["motor_targets"][1], np.clip(home + scale * 0.4, lo, hi),
                             atol=1e-6)  # late episode: previous action applied
  np.testing.assert_allclose(out.info["motor_targets"][3], np.clip(home, lo, hi),
                             atol=1e-6)  # on-time episode: current (zero) action applied
  assert out.metrics["debug/pushes"][1] == 1 and out.metrics["debug/pushes"][3] == 0
  np.testing.assert_allclose(out.info["push_vel"][1], kick1, rtol=1e-5)
  assert out.info["push_timer"][1] > 1  # next push scheduled
  # env 0 fell with dr_level 0 -> its new episode is the nominal robot, level kept
  assert out.info["dr_level"][0] == 0
  assert out.info["dr"]["friction"][0] == pytest.approx(1.0)
  np.testing.assert_allclose(out.info["dr"]["kp_scale"][0], 1.0, rtol=1e-6)
  assert int(out.info["dr"]["latency"][0]) == 0
  # The critic sees the randomisation of the current episode.
  priv_dr = out.priv_obs[:, env.priv_layout["dr"]]
  np.testing.assert_allclose(priv_dr[:, 0], out.info["dr"]["friction"], rtol=1e-6)


def test_set_dr_level(setup):
  env, _, s0 = setup
  s = env.set_dr_level(jax.device_get(s0), 0.3)
  assert np.allclose(s.info["dr_level"], 0.3) and s.info["dr_level"].shape == (N,)


def test_history_stacking_and_slope_gravity(setup):
  """v0.9: the policy obs = last 5 frames, newest first; the critic sees gravity along
  this episode's (possibly tilted) gravity direction."""
  from g1_rl import math as gm
  env, venv, s0 = setup
  F, H = env.frame_size, env.history
  s = jax.device_get(s0)
  for k in range(1, H):  # new episode: every slot holds the first frame
    np.testing.assert_allclose(s.obs[:, k * F:(k + 1) * F], s.obs[:, :F])
  out = jax.device_get(venv.step(jax.tree.map(jnp.copy, s0),
                                 parallel.put_batched(jnp.zeros((N, env.action_size)), RUNTIME.mesh)))
  live = out.done == 0
  np.testing.assert_allclose(out.obs[live, F:2 * F], s.obs[live, :F], atol=1e-6)  # shifted by one
  assert not np.allclose(out.obs[live, :F], s.obs[live, :F])
  # critic gravity = gravity direction of this episode, rotated into the body frame
  d = jax.vmap(env.packer.unpack)(out.data)
  want = jax.vmap(gm.quat_rotate_inverse)(d.qpos[:, 3:7], out.info["dr"]["gravity_dir"])
  np.testing.assert_allclose(out.priv_obs[:, env.priv_layout["obs"]][:, env.obs_layout["gravity"]],
                             np.asarray(want), atol=1e-5)
  tilt = np.degrees(np.arccos(-np.asarray(out.info["dr"]["gravity_dir"])[:, 2]))
  assert tilt.max() <= env.cfg.dr.slope_max_deg + 1e-3


def test_self_contacts_detects_crossed_legs(setup):
  from mujoco import mjx
  env, _, s0 = setup
  d = env.packer.unpack(jax.tree.map(lambda x: x[1], s0.data))
  count = jax.jit(lambda d: env._self_contacts(mjx.forward(env.mx, d)))
  assert float(count(d)) == 0  # home pose: no self contact
  names = env.robot.actuator_names
  q = d.qpos
  for side, sign in (("left", -1.0), ("right", 1.0)):  # swing both legs inward, across each other
    q = q.at[env._qadr[names.index(f"{side}_hip_roll_joint")]].set(sign * 0.5)
  assert float(count(d.replace(qpos=q))) > 0
