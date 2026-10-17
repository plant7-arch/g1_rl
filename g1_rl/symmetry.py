"""Left-right mirror symmetry of the G1 and of the policy's observations/actions.

Mirroring the robot about its sagittal (x-z) plane swaps every left joint with its
right counterpart. Joints that rotate about the body x or z axis (roll, yaw) also
change sign; joints about y (pitch, knee, elbow) keep it. In the base frame a vector
(x, y, z) becomes (x, -y, z) and an angular velocity (wx, wy, wz) becomes
(-wx, wy, -wz).

Each map is a (perm, sign) pair: mirrored = x[..., perm] * sign. Both maps are
involutions (applying them twice is the identity); tests/test_symmetry.py checks that
and that a mirrored joint configuration really is the mirror image in MuJoCo.

PPO uses `obs_map` / `action_map` for a mirror loss: the policy should act on a
mirrored observation with the mirrored action, so gaits come out left-right symmetric
and one-sided quirks (e.g. one arm held back) are discouraged.
"""

from __future__ import annotations

import numpy as np


def _partner(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[5:]
  if name.startswith("right_"):
    return "left_" + name[6:]
  return name


def joint_map(joint_names, joint_axes):
  """(perm, sign) over the actuated joints. `joint_axes` are the joints' local axes."""
  names = list(joint_names)
  perm = np.array([names.index(_partner(n)) for n in names])
  sign = np.array([1.0 if abs(ax[1]) > 0.5 else -1.0 for ax in np.asarray(joint_axes)])
  return perm, sign.astype(np.float32)


def obs_map(layout: dict, jperm, jsign):
  """(perm, sign) over the policy observation (layout = env.obs_layout)."""
  n = layout["_size"]
  perm, sign = np.arange(n), np.ones(n, np.float32)

  def put(name, p, s):
    sl = layout[name]
    perm[sl] = sl.start + np.asarray(p)
    sign[sl] = s

  put("gyro", [0, 1, 2], [-1, 1, -1])  # angular velocity (body frame)
  put("gravity", [0, 1, 2], [1, -1, 1])  # vector (body frame)
  put("command", [0, 1, 2], [1, -1, -1])  # vx, vy, wz
  for k in ("joint_pos", "joint_vel", "last_act"):
    put(k, jperm, jsign)
  put("gait", [1, 0, 3, 2], 1)  # [cos_l, cos_r, sin_l, sin_r]: the feet swap clocks
  return perm, sign


def for_env(env):
  """(obs_perm, obs_sign, act_perm, act_sign) for a G1Env."""
  m = env.mj_model
  axes = m.jnt_axis[m.actuator_trnid[:, 0]]
  jp, js = joint_map(env.robot.actuator_names, axes)
  op, os_ = obs_map(env.obs_layout, jp, js)
  return op, os_, jp, js


def apply(x, perm, sign):
  return x[..., perm] * sign
