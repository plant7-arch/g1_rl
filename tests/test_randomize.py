"""Phase 3: domain randomisation sampling and model application (no simulation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx

from g1_rl import randomize, robot
from g1_rl.config import DomainRandConfig, SimConfig

CFG = DomainRandConfig()


@pytest.fixture(scope="module")
def setup():
  m = robot.load_mj_model()
  info = robot.robot_info(m)
  r = randomize.Randomizer(m, CFG, info.torso_body_id, SimConfig().ctrl_dt)
  return m, info, r, mjx.put_model(m)


MODEL_FIELDS = ("pair_friction", "body_mass", "body_inertia", "body_ipos", "actuator_gainprm",
                "actuator_biasprm", "dof_frictionloss", "dof_armature")


def test_level_zero_is_the_nominal_robot(setup):
  m, _, r, mx = setup
  out = r.apply(mx, r.sample(jax.random.PRNGKey(3), 0.0))
  for f in MODEL_FIELDS:
    np.testing.assert_allclose(np.asarray(getattr(out, f)), np.asarray(getattr(mx, f)),
                               rtol=1e-6, atol=1e-7, err_msg=f)
  assert not bool(r.sample(jax.random.PRNGKey(3), 0.0)["latency"])


def test_full_level_samples_within_ranges(setup):
  _, _, r, _ = setup
  dr = jax.vmap(lambda k: r.sample(k, 1.0))(jax.random.split(jax.random.PRNGKey(0), 256))
  checks = {"friction": CFG.friction, "mass_scale": CFG.link_mass_scale,
            "payload": CFG.torso_payload, "kp_scale": CFG.kp_scale, "kd_scale": CFG.kd_scale,
            "frictionloss_scale": CFG.frictionloss_scale, "armature_scale": CFG.armature_scale}
  for k, (lo, hi) in checks.items():
    v = np.asarray(dr[k])
    assert v.min() >= lo - 1e-6 and v.max() <= hi + 1e-6, k
    assert v.std() > 0.1 * (hi - lo) / np.sqrt(12), f"{k} barely varies"
  assert np.abs(np.asarray(dr["com_offset"])).max() <= CFG.torso_com_offset + 1e-6
  lat = np.asarray(dr["latency"]).mean()
  assert abs(lat - CFG.latency_prob) < 0.12
  # per-actuator gains vary within one robot
  assert np.asarray(dr["kp_scale"])[0].std() > 0


def test_level_scales_toward_nominal(setup):
  _, _, r, _ = setup
  key = jax.random.PRNGKey(7)
  full, half = r.sample(key, 1.0), r.sample(key, 0.5)
  for k, nominal in [("friction", 1.0), ("kp_scale", 1.0), ("payload", 0.0), ("com_offset", 0.0)]:
    np.testing.assert_allclose(np.asarray(half[k]) - nominal, 0.5 * (np.asarray(full[k]) - nominal),
                               rtol=1e-5, atol=1e-7, err_msg=k)
  timer, kick = r.sample_push(key, 0.5)
  _, kick_full = r.sample_push(key, 1.0)
  np.testing.assert_allclose(np.asarray(kick), 0.5 * np.asarray(kick_full), rtol=1e-5)
  lo, hi = CFG.push_interval
  assert lo / 0.02 - 1 <= int(timer) <= hi / 0.02


def test_apply_writes_the_right_fields(setup):
  m, info, r, mx = setup
  dr = r.nominal()
  dr.update(friction=jnp.asarray(0.5), payload=jnp.asarray(2.0),
            com_offset=jnp.array([0.01, 0.0, -0.02]),
            kp_scale=jnp.full(m.nu, 1.1), kd_scale=jnp.full(m.nu, 0.8))
  out = r.apply(mx, dr)
  t = info.torso_body_id
  assert float(out.body_mass[t]) == pytest.approx(m.body_mass[t] + 2.0, rel=1e-6)
  assert float(out.body_mass.sum()) == pytest.approx(m.body_mass.sum() + 2.0, rel=1e-6)
  np.testing.assert_allclose(np.asarray(out.body_ipos[t]), m.body_ipos[t] + [0.01, 0, -0.02], atol=1e-7)
  # Position actuator: force = kp*(target - q) - kd*qdot  <=>  gain0 = kp, bias1 = -kp, bias2 = -kd
  gain, bias = np.asarray(out.actuator_gainprm), np.asarray(out.actuator_biasprm)
  np.testing.assert_allclose(gain[:, 0], m.actuator_gainprm[:, 0] * 1.1, rtol=1e-6)
  np.testing.assert_allclose(bias[:, 1], -gain[:, 0], rtol=1e-6)
  np.testing.assert_allclose(bias[:, 2], m.actuator_biasprm[:, 2] * 0.8, rtol=1e-6)
  # Friction only on floor contacts; robot self-collision pairs untouched.
  fr = np.asarray(out.pair_friction)
  floor = np.zeros(m.npair, bool)
  floor[r.floor_pairs] = True
  assert np.allclose(fr[floor, :2], 0.5) and np.allclose(fr[~floor], m.pair_friction[~floor])
  assert floor.sum() >= 6 and (~floor).sum() >= 1


def test_privileged_block_size(setup):
  _, _, r, _ = setup
  assert r.privileged(r.nominal(), jnp.zeros(2)).shape == (r.priv_size,)
