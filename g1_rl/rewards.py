"""Reward terms for velocity-command walking.

Every term is a pure function of a `RewardInputs` (plain arrays for ONE env,
built by the env) and `RewardConsts` (per-robot constants). That keeps the
terms easy to read and to unit-test with hand-made numbers (tests/test_rewards.py).

`compute` returns each term already multiplied by its weight and by ctrl_dt, so
the per-term metrics logged during training are directly comparable shares of
the return. `total` sums them, optionally clips at 0, then adds termination.
Add a term: write `_my_term(i, c, cfg) -> scalar`, register it in TERMS and give
it a weight in config.RewardConfig.weights (weight 0 disables it).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class RewardInputs(NamedTuple):
  command: jax.Array  # (3,) vx, vy, wz
  lin_vel: jax.Array  # (3,) base linear velocity in the heading (yaw-only) frame
  ang_vel: jax.Array  # (3,) base angular velocity, body frame
  gravity: jax.Array  # (3,) gravity direction in the body frame
  base_height: jax.Array  # ()
  joint_pos: jax.Array  # (nu,)
  joint_vel: jax.Array  # (nu,)
  joint_vel_prev: jax.Array  # (nu,)
  torques: jax.Array  # (nu,) actuator forces
  action: jax.Array  # (nu,)
  last_action: jax.Array  # (nu,)
  foot_height: jax.Array  # (2,) lowest point of each foot above the floor
  foot_vel_xy: jax.Array  # (2, 2) horizontal foot velocity (world)
  foot_up_z: jax.Array  # (2,) z-component of each sole's normal (1 = flat)
  contact: jax.Array  # (2,) bool
  first_contact: jax.Array  # (2,) bool, touchdown this step
  air_time: jax.Array  # (2,) seconds in the air, including this step
  phase: jax.Array  # (2,) gait clock per foot in [-pi, pi); swing when sin > 0
  feet_lateral_distance: jax.Array  # () sideways distance between feet, heading frame
  self_contacts: jax.Array  # () robot-robot contact pairs in penetration
  fell: jax.Array  # () bool


class RewardConsts(NamedTuple):
  home: np.ndarray  # (nu,) joint targets of the home pose
  soft_lo: np.ndarray  # (nu,)
  soft_hi: np.ndarray  # (nu,)
  pose_weights: np.ndarray  # (nu,)
  arm_mask: np.ndarray  # (nu,) 1 for shoulder / elbow / wrist joints
  dt: float


ARM_JOINTS = ("shoulder", "elbow", "wrist")


def make_consts(joint_names, home, lo, hi, cfg, dt) -> RewardConsts:
  mid, half = (hi + lo) / 2, (hi - lo) / 2 * cfg.soft_joint_limit
  w = np.zeros(len(joint_names), np.float32)
  for i, n in enumerate(joint_names):
    hits = [v for k, v in cfg.pose_weights.items() if k in n]
    if not hits:
      raise KeyError(f"no pose weight matches joint {n!r}")
    w[i] = hits[0]
  arm = np.array([any(k in n for k in ARM_JOINTS) for n in joint_names], np.float32)
  return RewardConsts(np.asarray(home, np.float32), (mid - half).astype(np.float32),
                      (mid + half).astype(np.float32), w, arm, float(dt))


def _moving(i):
  """1 when the command asks for motion."""
  return (jnp.linalg.norm(i.command) > 1e-3).astype(jnp.float32)


def _stepping(i, cfg):
  """1 when the feet should follow the gait clock: a motion command, OR the base is
  sliding (e.g. after a push) while asked to stand. Without the second case the
  stand-still rewards forbid the recovery step a pushed robot needs."""
  disturbed = jnp.linalg.norm(i.lin_vel[:2]) > cfg.stand_recovery_speed
  return jnp.maximum(_moving(i), disturbed.astype(jnp.float32))


# --- task -----------------------------------------------------------------------
def _tracking_lin_vel(i, c, cfg):
  err = jnp.sum(jnp.square(i.command[:2] - i.lin_vel[:2]))
  return jnp.exp(-err / cfg.tracking_sigma)


def _tracking_lin_vel_fine(i, c, cfg):
  err = jnp.sum(jnp.square(i.command[:2] - i.lin_vel[:2]))
  return jnp.exp(-err / cfg.tracking_fine_sigma)


def _tracking_ang_vel_fine(i, c, cfg):
  return jnp.exp(-jnp.square(i.command[2] - i.ang_vel[2]) / cfg.tracking_fine_sigma)


def _tracking_ang_vel(i, c, cfg):
  return jnp.exp(-jnp.square(i.command[2] - i.ang_vel[2]) / cfg.tracking_ang_sigma)


def _alive(i, c, cfg):
  return jnp.ones(())


# --- gait -----------------------------------------------------------------------
def _swing(i, cfg):
  """Desired swing indicator per foot: 1 in the swing half of the clock while stepping."""
  return (jnp.sin(i.phase) > 0).astype(jnp.float32) * _stepping(i, cfg)


def _feet_phase(i, c, cfg):
  target = cfg.swing_height * jnp.maximum(jnp.sin(i.phase), 0.0) * _stepping(i, cfg)
  err = jnp.sum(jnp.square(i.foot_height - target))
  return jnp.exp(-err / cfg.feet_phase_sigma)


def _gait_contact(i, c, cfg):
  want_contact = _swing(i, cfg) < 0.5
  return jnp.mean((i.contact == want_contact).astype(jnp.float32))


def _feet_air_time(i, c, cfg):
  bonus = jnp.minimum(i.air_time - cfg.air_time_target, cfg.air_time_max_bonus)
  return jnp.sum(bonus * i.first_contact) * _moving(i)


def _feet_slip(i, c, cfg):
  return jnp.sum(jnp.sum(jnp.square(i.foot_vel_xy), -1) * i.contact)


def _feet_flat(i, c, cfg):
  return jnp.sum((1.0 - i.foot_up_z) * i.contact)


def _feet_distance(i, c, cfg):
  return jnp.maximum(cfg.min_feet_distance - i.feet_lateral_distance, 0.0)


def _self_collision(i, c, cfg):
  return i.self_contacts


# --- base -----------------------------------------------------------------------
def _lin_vel_z(i, c, cfg):
  return jnp.square(i.lin_vel[2])


def _ang_vel_xy(i, c, cfg):
  return jnp.sum(jnp.square(i.ang_vel[:2]))


def _orientation(i, c, cfg):
  return jnp.sum(jnp.square(i.gravity[:2]))


def _base_height(i, c, cfg):
  return jnp.square(i.base_height - cfg.base_height_target)


# --- posture / regularisation ------------------------------------------------------
def _pose(i, c, cfg):
  return jnp.sum(c.pose_weights * jnp.square(i.joint_pos - c.home))


def _stand_still(i, c, cfg):
  return jnp.sum(jnp.abs(i.joint_pos - c.home)) * (1.0 - _stepping(i, cfg))


def _arm_still(i, c, cfg):
  """Arms near home while standing (no command, base still). Walking leaves them to the
  much weaker `pose` term so they can swing."""
  return jnp.sum(c.arm_mask * jnp.square(i.joint_pos - c.home)) * (1.0 - _stepping(i, cfg))


def _joint_limits(i, c, cfg):
  return jnp.sum(jnp.maximum(i.joint_pos - c.soft_hi, 0.0) + jnp.maximum(c.soft_lo - i.joint_pos, 0.0))


def _joint_vel(i, c, cfg):
  return jnp.sum(jnp.square(i.joint_vel))


def _joint_acc(i, c, cfg):
  return jnp.sum(jnp.square((i.joint_vel - i.joint_vel_prev) / c.dt))


def _torques(i, c, cfg):
  return jnp.sum(jnp.square(i.torques))


def _energy(i, c, cfg):
  return jnp.sum(jnp.abs(i.torques * i.joint_vel))


def _action_rate(i, c, cfg):
  return jnp.sum(jnp.square(i.action - i.last_action))


def _termination(i, c, cfg):
  return i.fell.astype(jnp.float32)


TERMS = {
    "tracking_lin_vel": _tracking_lin_vel, "tracking_ang_vel": _tracking_ang_vel,
    "tracking_lin_vel_fine": _tracking_lin_vel_fine, "tracking_ang_vel_fine": _tracking_ang_vel_fine,
    "alive": _alive, "feet_phase": _feet_phase, "gait_contact": _gait_contact,
    "feet_air_time": _feet_air_time, "feet_slip": _feet_slip, "feet_flat": _feet_flat,
    "feet_distance": _feet_distance, "self_collision": _self_collision,
    "lin_vel_z": _lin_vel_z, "ang_vel_xy": _ang_vel_xy,
    "orientation": _orientation, "base_height": _base_height, "pose": _pose,
    "stand_still": _stand_still, "arm_still": _arm_still, "joint_limits": _joint_limits,
    "joint_vel": _joint_vel,
    "joint_acc": _joint_acc, "torques": _torques, "energy": _energy,
    "action_rate": _action_rate, "termination": _termination,
}
POST_CLIP = ("termination",)


def compute(i: RewardInputs, c: RewardConsts, cfg) -> dict:
  """Weighted, dt-scaled value of every enabled term (dict name -> scalar)."""
  unknown = set(cfg.weights) - set(TERMS)
  if unknown:
    raise KeyError(f"reward weights for unknown terms: {sorted(unknown)}")
  out = {}
  for name, w in cfg.weights.items():
    if w == 0:
      continue
    v = TERMS[name](i, c, cfg) * (w * c.dt)
    out[name] = jnp.where(jnp.isfinite(v), v, 0.0)
  return out


def pre_clip(terms: dict) -> jax.Array:
  return sum((v for k, v in terms.items() if k not in POST_CLIP), jnp.zeros(()))


def total(terms: dict, cfg) -> jax.Array:
  pre = pre_clip(terms)
  if cfg.only_positive:
    pre = jnp.maximum(pre, 0.0)
  return pre + sum((v for k, v in terms.items() if k in POST_CLIP), jnp.zeros(()))
