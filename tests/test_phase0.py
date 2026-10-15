"""Phase 0 tests: model constants, MJX correctness vs. MuJoCo, data packing, sharding."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from conftest import RUNTIME
from g1_rl import parallel, robot, sim
from g1_rl.config import SimConfig

CFG = SimConfig()


@pytest.fixture(scope="module")
def model():
  m = robot.load_mj_model(CFG)
  return m, robot.robot_info(m, CFG.home_key), mjx.put_model(m, impl=CFG.impl)


def test_robot_info(model):
  m, info, _ = model
  assert info.nu == 29 and m.nq == 36 and m.nv == 35
  groups = np.concatenate([info.leg_idx, info.waist_idx, info.arm_idx])
  assert sorted(groups.tolist()) == list(range(29))  # groups partition the actuators
  assert np.all(info.home_ctrl >= info.ctrl_lo) and np.all(info.home_ctrl <= info.ctrl_hi)
  assert CFG.n_substeps == 5


def test_packer_roundtrip(model):
  _, info, mx = model
  pk = sim.DataPacker(mx)
  d = sim.init_data(mx, jnp.asarray(info.home_qpos), jnp.asarray(info.home_ctrl))
  d2 = pk.unpack(pk.pack(d))
  for a, b in zip(jax.tree.leaves(d), jax.tree.leaves(d2)):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
  assert all(l.size > 0 for l in pk.pack(d))


def test_mjx_matches_mujoco(model):
  m, info, mx = model
  ref = mujoco.MjData(m)
  ref.qpos[:], ref.ctrl[:] = info.home_qpos, info.home_ctrl
  mujoco.mj_forward(m, ref)
  for _ in range(50):
    mujoco.mj_step(m, ref)
  init = jax.jit(lambda q, u: sim.init_data(mx, q, u))
  step = jax.jit(lambda d: sim.step_ctrl(mx, d, d.ctrl, CFG.n_substeps))
  d = init(jnp.asarray(info.home_qpos), jnp.asarray(info.home_ctrl))
  for _ in range(10):
    d = step(d)
  q = np.asarray(d.qpos)
  assert np.isfinite(q).all()
  np.testing.assert_allclose(q[:3], ref.qpos[:3], atol=0.02)
  np.testing.assert_allclose(q[7:], ref.qpos[7:], atol=0.05)


def test_sharded_batch_step(model):
  _, info, mx = model
  pk = sim.DataPacker(mx)
  n = 2 * RUNTIME.n_devices
  home_q, home_u = jnp.asarray(info.home_qpos), jnp.asarray(info.home_ctrl)

  def init_one(key):
    q = home_q.at[7:].add(jax.random.uniform(key, (29,), minval=-0.02, maxval=0.02))
    return pk.pack(sim.init_data(mx, q, home_u))

  def step_one(p):
    return pk.pack(sim.step_ctrl(mx, pk.unpack(p), home_u, CFG.n_substeps))

  keys = parallel.put_batched(jax.random.split(jax.random.PRNGKey(0), n), RUNTIME.mesh)
  p = parallel.jit_batched(jax.vmap(init_one), RUNTIME.mesh)(keys)
  p = parallel.jit_batched(jax.vmap(step_one), RUNTIME.mesh)(p)
  d = jax.vmap(pk.unpack)(p)
  assert d.qpos.shape == (n, 36)
  assert bool(jnp.isfinite(d.qpos).all())
  # Robots started from different poses must evolve differently.
  assert not np.allclose(np.asarray(d.qpos[0]), np.asarray(d.qpos[1]))
