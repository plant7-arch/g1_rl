"""Domain randomisation: per-episode physics parameters, action latency and pushes.

Parameters are sampled once per episode (at reset) into a small dict that lives
in the env state (info["dr"]). Every step, `apply` writes them into a copy of the
MJX model; under vmap each env therefore simulates its own robot. Everything is
expressed relative to the nominal model, and `level` in [0, 1] shrinks every
range toward nominal (curriculum).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from g1_rl.config import DomainRandConfig


def latency_steps(u, probs, level):
  """0, 1 or 2 control steps of delay from one uniform draw; `level` scales the odds."""
  p1, p2 = probs[0] * level, probs[1] * level
  return (u < p1 + p2).astype(jnp.int32) + (u < p2).astype(jnp.int32)


def slope_gravity(k_on, k_tilt, k_dir, prob, max_deg, level):
  """Unit gravity direction for this episode: straight down, or tilted by up to
  `max_deg` (times level) in a random direction, which is the same as standing on an
  endless incline of that angle."""
  on = jax.random.uniform(k_on) < prob
  tilt = jax.random.uniform(k_tilt) * jnp.deg2rad(max_deg) * level * on
  ang = jax.random.uniform(k_dir, minval=-jnp.pi, maxval=jnp.pi)
  return jnp.array([jnp.sin(tilt) * jnp.cos(ang), jnp.sin(tilt) * jnp.sin(ang), -jnp.cos(tilt)])


class Randomizer:

  def __init__(self, mj_model, cfg: DomainRandConfig, torso_body_id: int, ctrl_dt: float):
    m = mj_model
    self.cfg = cfg
    self.ctrl_dt = ctrl_dt
    self.nbody, self.nv, self.nu = m.nbody, m.nv, m.nu
    self.torso = int(torso_body_id)
    self.floor_pairs = np.array([i for i in range(m.npair) if m.pair(i).name.endswith("_floor")],
                                np.int32)
    if len(self.floor_pairs) == 0:
      raise ValueError("no '*_floor' contact pairs found in the model")
    self.kp0 = m.actuator_gainprm[:, 0].astype(np.float32)
    self.kd0 = (-m.actuator_biasprm[:, 2]).astype(np.float32)
    # Size of the privileged-observation block describing this episode's robot:
    # friction, payload, CoM offset (3), kp (nu), kd (nu), latency, recent push (2),
    # slope = gravity direction x, y (2), IMU tilt (2), gyro bias (3).
    self.priv_size = 1 + 1 + 3 + self.nu + self.nu + 1 + 2 + 2 + 2 + 3

  # ------------------------------------------------------------------ sampling
  def sample(self, rng, level) -> dict:
    c = self.cfg
    k = jax.random.split(rng, 16)

    def scaled(key, rng_range, shape=(), nominal=1.0):
      x = jax.random.uniform(key, shape, minval=rng_range[0], maxval=rng_range[1])
      return nominal + level * (x - nominal)

    return {
        "friction": scaled(k[0], c.friction),
        "mass_scale": scaled(k[1], c.link_mass_scale, (self.nbody,)),
        "payload": scaled(k[2], c.torso_payload, nominal=0.0),
        "com_offset": level * jax.random.uniform(
            k[3], (3,), minval=-c.torso_com_offset, maxval=c.torso_com_offset),
        "kp_scale": scaled(k[4], c.kp_scale, (self.nu,)),
        "kd_scale": scaled(k[5], c.kd_scale, (self.nu,)),
        "frictionloss_scale": scaled(k[6], c.frictionloss_scale, (self.nv,)),
        "armature_scale": scaled(k[7], c.armature_scale, (self.nv,)),
        # control steps of action delay: 0, 1 or 2
        "latency": latency_steps(jax.random.uniform(k[8]), c.latency_probs, level),
        "gravity_dir": slope_gravity(k[9], k[10], k[11], c.slope_prob, c.slope_max_deg, level),
        "imu_tilt": level * jax.random.uniform(
            k[12], (2,), minval=-1.0, maxval=1.0) * jnp.deg2rad(c.imu_tilt_deg),
        "gyro_bias": level * jax.random.uniform(k[13], (3,), minval=-c.gyro_bias, maxval=c.gyro_bias),
        "joint_offset": level * jax.random.uniform(
            k[14], (self.nu,), minval=-c.joint_offset, maxval=c.joint_offset),
    }

  def nominal(self) -> dict:
    return self.sample(jax.random.PRNGKey(0), 0.0)

  def sample_push(self, rng, level):
    """(timer in control steps, horizontal velocity kick) for the next push."""
    c = self.cfg
    k1, k2, k3 = jax.random.split(rng, 3)
    steps = jax.random.uniform(k1, minval=c.push_interval[0], maxval=c.push_interval[1]) / self.ctrl_dt
    ang = jax.random.uniform(k2, minval=-jnp.pi, maxval=jnp.pi)
    mag = jax.random.uniform(k3, minval=c.push_velocity[0], maxval=c.push_velocity[1]) * level
    return steps.astype(jnp.int32), mag * jnp.array([jnp.cos(ang), jnp.sin(ang)])

  # ------------------------------------------------------------------ applying
  def apply(self, mx, dr: dict):
    """Model with this env's parameters (call inside the per-env, vmapped step)."""
    fr = mx.pair_friction.at[self.floor_pairs, 0].set(dr["friction"])
    fr = fr.at[self.floor_pairs, 1].set(dr["friction"])
    mass = mx.body_mass * dr["mass_scale"]
    inertia = mx.body_inertia * dr["mass_scale"][:, None]
    torso_m = mass[self.torso]
    new_torso_m = jnp.maximum(torso_m + dr["payload"], 0.1 * torso_m)
    inertia = inertia.at[self.torso].multiply(new_torso_m / torso_m)
    mass = mass.at[self.torso].set(new_torso_m)
    ipos = mx.body_ipos.at[self.torso].add(dr["com_offset"])
    kp = self.kp0 * dr["kp_scale"]
    kd = self.kd0 * dr["kd_scale"]
    gain = mx.actuator_gainprm.at[:, 0].set(kp)
    bias = mx.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)
    gravity = 9.81 * dr["gravity_dir"]
    return mx.replace(
        opt=mx.opt.replace(gravity=gravity),
        pair_friction=fr, body_mass=mass, body_inertia=inertia, body_ipos=ipos,
        actuator_gainprm=gain, actuator_biasprm=bias,
        dof_frictionloss=mx.dof_frictionloss * dr["frictionloss_scale"],
        dof_armature=mx.dof_armature * dr["armature_scale"])

  def privileged(self, dr: dict, push_vel) -> jax.Array:
    return jnp.concatenate([
        dr["friction"][None], dr["payload"][None], dr["com_offset"] / max(self.cfg.torso_com_offset, 1e-6),
        dr["kp_scale"], dr["kd_scale"], dr["latency"].astype(jnp.float32)[None], push_vel,
        dr["gravity_dir"][:2] / max(np.sin(np.deg2rad(self.cfg.slope_max_deg)), 1e-6),
        dr["imu_tilt"] / max(np.deg2rad(self.cfg.imu_tilt_deg), 1e-6),
        dr["gyro_bias"] / max(self.cfg.gyro_bias, 1e-6)])
