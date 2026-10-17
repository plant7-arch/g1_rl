"""Mirror maps (g1_rl/symmetry.py): involutions, true mirror images in MuJoCo, and the
policy observation of a mirrored robot equals the mirrored observation."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from conftest import RUNTIME  # noqa: F401  (sets up the platform)

from g1_rl import robot, symmetry
from g1_rl.config import EnvConfig, NoiseConfig
from g1_rl.env import G1Env


@pytest.fixture(scope="module")
def env():
  return G1Env(EnvConfig(noise=NoiseConfig(level=0.0)))


def test_maps_are_involutions(env):
  op, os_, ap, as_ = symmetry.for_env(env)
  x = np.random.default_rng(0).normal(size=(5, env.obs_size))
  np.testing.assert_allclose(symmetry.apply(symmetry.apply(x, op, os_), op, os_), x)
  a = np.random.default_rng(1).normal(size=(5, env.action_size))
  np.testing.assert_allclose(symmetry.apply(symmetry.apply(a, ap, as_), ap, as_), a)
  names = env.robot.actuator_names
  assert names[ap[names.index("left_hip_roll_joint")]] == "right_hip_roll_joint"
  assert as_[names.index("left_hip_roll_joint")] == -1 and as_[names.index("left_knee_joint")] == 1
  assert ap[names.index("waist_yaw_joint")] == names.index("waist_yaw_joint")


def test_mirrored_joints_give_mirror_image(env):
  m = env.mj_model
  d = mujoco.MjData(m)
  info = env.robot
  _, _, jp, js = symmetry.for_env(env)
  bodies = [m.body(i).name for i in range(m.nbody)]
  bperm = [bodies.index(symmetry._partner(n)) for n in bodies]

  def xpos(q):
    d.qpos[:] = 0
    d.qpos[2], d.qpos[3] = 1.0, 1.0
    d.qpos[info.actuator_qpos_adr] = q
    mujoco.mj_kinematics(m, d)
    return d.xpos.copy()

  rng = np.random.default_rng(0)
  for _ in range(10):
    q = rng.uniform(np.maximum(info.ctrl_lo, -1), np.minimum(info.ctrl_hi, 1))
    q = np.clip(q[jp] * js, info.ctrl_lo, info.ctrl_hi)[jp] * js  # both q and mirror(q) in range
    p, pm = xpos(q), xpos(q[jp] * js)
    np.testing.assert_allclose(pm, p[bperm] * np.array([1, -1, 1]), atol=1e-4)
  # The home pose is symmetric.
  np.testing.assert_allclose(info.home_ctrl[jp] * js, info.home_ctrl, atol=1e-6)


def test_observation_of_mirrored_robot(env):
  """Build a random state, mirror the robot about the world x-z plane, and compare
  the policy observations (noise off)."""
  op, os_, jp, js = symmetry.for_env(env)
  s = jax.jit(env.reset)(jax.random.PRNGKey(3), 0.0)
  d = env.packer.unpack(s.data)
  rng = np.random.default_rng(2)
  nu = env.nu
  qa, va = env.robot.actuator_qpos_adr, env.robot.actuator_qvel_adr
  q = np.asarray(d.qpos).copy()
  q[qa] += rng.uniform(-0.2, 0.2, nu)
  q[3:7] = np.array([0.97, 0.1, -0.12, 0.17]) / np.linalg.norm([0.97, 0.1, -0.12, 0.17])
  v = rng.normal(size=env.mx.nv) * 0.5
  info = dict(s.info)
  info["command"] = jnp.array([0.4, 0.2, -0.5])
  info["last_act"] = jnp.asarray(rng.uniform(-1, 1, nu))
  info["phase"] = jnp.array([0.7, 0.7 - np.pi])

  qm = q.copy()
  qm[1] = -q[1]
  qm[3:7] = q[3:7] * np.array([1, -1, 1, -1])
  qm[qa] = q[qa][jp] * js
  vm = v.copy()
  vm[0:3] = v[0:3] * np.array([1, -1, 1])
  vm[3:6] = v[3:6] * np.array([-1, 1, -1])
  vm[va] = v[va][jp] * js
  info_m = dict(info)
  info_m["command"] = info["command"] * jnp.array([1, -1, -1])
  info_m["last_act"] = info["last_act"][jp] * js
  info_m["phase"] = info["phase"][::-1]

  @jax.jit
  def observe(q, v, info):
    from mujoco import mjx
    dd = mjx.kinematics(env.mx, d.replace(qpos=q, qvel=v))
    return env._observe(dd, info, jax.random.PRNGKey(0))[0]

  o = np.asarray(observe(jnp.asarray(q), jnp.asarray(v), info))
  om = np.asarray(observe(jnp.asarray(qm), jnp.asarray(vm), info_m))
  np.testing.assert_allclose(om, symmetry.apply(o, op, os_), atol=1e-5)
