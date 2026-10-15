"""Unitree G1 model loading and constants extracted from the MJCF (not hard-coded).

Everything here is host-side (NumPy / classic MuJoCo). Device placement happens
in ``make_mjx_model``.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from g1_rl.config import SimConfig

# Joint groups by name prefix; used for per-group reward weights / noise later.
LEG_JOINTS = tuple(
    f"{s}_{j}_joint" for s in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINTS = tuple(
    f"{s}_{j}_joint" for s in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"))
FEET_SITES = ("left_foot", "right_foot")
ROOT_BODY = "pelvis"
TORSO_BODY = "torso_link"
IMU_SITE = "imu_in_pelvis"


@dataclasses.dataclass(frozen=True)
class RobotInfo:
  nq: int
  nv: int
  nu: int
  actuator_names: tuple
  actuator_qpos_adr: np.ndarray  # qpos index of each actuated joint
  actuator_qvel_adr: np.ndarray  # qvel index of each actuated joint
  home_qpos: np.ndarray  # (nq,)
  home_ctrl: np.ndarray  # (nu,) joint targets of the home pose
  ctrl_lo: np.ndarray
  ctrl_hi: np.ndarray
  leg_idx: np.ndarray  # indices into the actuator vector
  waist_idx: np.ndarray
  arm_idx: np.ndarray
  feet_site_ids: np.ndarray
  imu_site_id: int
  root_body_id: int
  torso_body_id: int
  total_mass: float


def load_mj_model(cfg: SimConfig = SimConfig(), xml_path: str | None = None) -> mujoco.MjModel:
  m = mujoco.MjModel.from_xml_path(xml_path or cfg.xml_path)
  if abs(m.opt.timestep - cfg.sim_dt) > 1e-12:
    m.opt.timestep = cfg.sim_dt
  if cfg.iterations is not None:
    m.opt.iterations = cfg.iterations
  if cfg.ls_iterations is not None:
    m.opt.ls_iterations = cfg.ls_iterations
  return m


def robot_info(m: mujoco.MjModel, home_key: str = "knees_bent") -> RobotInfo:
  names = tuple(m.actuator(i).name for i in range(m.nu))
  jnt_ids = m.actuator_trnid[:, 0]
  qpos_adr = m.jnt_qposadr[jnt_ids].astype(np.int32)
  qvel_adr = m.jnt_dofadr[jnt_ids].astype(np.int32)

  key = m.key(home_key)
  home_qpos = np.array(key.qpos, dtype=np.float32)
  home_ctrl = home_qpos[qpos_adr].copy()

  def idx(group):
    missing = [g for g in group if g not in names]
    if missing:
      raise KeyError(f"Actuators missing from model: {missing}")
    return np.array([names.index(g) for g in group], dtype=np.int32)

  return RobotInfo(
      nq=m.nq, nv=m.nv, nu=m.nu,
      actuator_names=names,
      actuator_qpos_adr=qpos_adr,
      actuator_qvel_adr=qvel_adr,
      home_qpos=home_qpos,
      home_ctrl=home_ctrl,
      ctrl_lo=m.actuator_ctrlrange[:, 0].astype(np.float32),
      ctrl_hi=m.actuator_ctrlrange[:, 1].astype(np.float32),
      leg_idx=idx(LEG_JOINTS),
      waist_idx=idx(WAIST_JOINTS),
      arm_idx=idx(ARM_JOINTS),
      feet_site_ids=np.array([m.site(s).id for s in FEET_SITES], dtype=np.int32),
      imu_site_id=m.site(IMU_SITE).id,
      root_body_id=m.body(ROOT_BODY).id,
      torso_body_id=m.body(TORSO_BODY).id,
      total_mass=float(m.body_subtreemass[m.body(ROOT_BODY).id]),
  )


def make_mjx_model(m: mujoco.MjModel, cfg: SimConfig = SimConfig()):
  """Put the model on the default device. Under shard_map it is replicated."""
  from mujoco import mjx  # noqa: PLC0415
  return mjx.put_model(m, impl=cfg.impl)


def summary(m: mujoco.MjModel, info: RobotInfo) -> str:
  return (f"G1: nq={m.nq} nv={m.nv} nu={m.nu} bodies={m.nbody} geoms={m.ngeom} "
          f"contact_pairs={m.npair} mass={info.total_mass:.1f}kg "
          f"dt={m.opt.timestep} solver_iters={m.opt.iterations}/{m.opt.ls_iterations}")
