"""Central configuration. Later phases add EnvConfig / RewardConfig / PPOConfig here."""

from __future__ import annotations

import dataclasses
import os

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PACKAGE_ROOT, "assets")


@dataclasses.dataclass(frozen=True)
class SimConfig:
  # Training scene: simplified collision geometry + explicit contact pairs.
  xml_path: str = os.path.join(ASSETS_DIR, "scene_mjx.xml")
  # Full-mesh scene used only for sim-to-sim evaluation (classic MuJoCo).
  eval_xml_path: str = os.path.join(ASSETS_DIR, "scene.xml")
  sim_dt: float = 0.004  # matches <option timestep> in g1_mjx.xml
  ctrl_dt: float = 0.02  # policy runs at 50 Hz
  # MJX backend: "jax" (portable, default). "warp" is optional, needs mujoco_warp.
  impl: str = "jax"
  # Solver overrides (None = keep the XML values: iterations=5, ls_iterations=8).
  iterations: int | None = None
  ls_iterations: int | None = None
  home_key: str = "knees_bent"
  action_scale: float = 0.25

  @property
  def n_substeps(self) -> int:
    n = round(self.ctrl_dt / self.sim_dt)
    assert abs(n * self.sim_dt - self.ctrl_dt) < 1e-9, "ctrl_dt must be a multiple of sim_dt"
    return n


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
  """Uniform observation noise, half-widths (scaled by `level`). Policy obs only."""
  level: float = 1.0
  gyro: float = 0.2  # rad/s
  gravity: float = 0.05
  joint_pos: float = 0.03  # rad
  joint_vel: float = 1.5  # rad/s


@dataclasses.dataclass(frozen=True)
class CommandConfig:
  lin_vel_x: tuple = (-0.6, 1.0)  # m/s, forward
  lin_vel_y: tuple = (-0.5, 0.5)  # m/s, left
  ang_vel_z: tuple = (-1.0, 1.0)  # rad/s, yaw
  zero_prob: float = 0.1  # fraction of commands that are "stand still"
  deadband: float = 0.1  # |component| below this is snapped to 0
  resample_steps: tuple = (250, 500)  # resample every 5-10 s (at 50 Hz)


def _default_weights():
  # Every term is multiplied by its weight and by ctrl_dt (0.02 s), so a weight of
  # 1.0 held for a whole 20 s episode adds 20 to the return. Positive = reward.
  return {
      # task
      "tracking_lin_vel": 1.5,  # exp(-|v_xy_cmd - v_xy|^2 / sigma), heading frame
      "tracking_ang_vel": 1.0,  # (v0.7: was 0.75) exp(-(wz_cmd - wz)^2 / sigma)
      "alive": 0.25,
      # gait shaping (uses the gait clock the policy observes)
      "feet_phase": 1.0,  # swing-foot height follows the clock
      "gait_contact": 0.5,  # stance/swing pattern matches the clock
      "feet_air_time": 1.0,  # reward steps of a reasonable duration (on touchdown)
      "feet_slip": -0.25,  # horizontal speed of feet in contact
      "feet_flat": -2.0,  # tilted soles while in contact
      "feet_distance": -10.0,  # feet closer than min_feet_distance (sideways)
      # base stability
      "lin_vel_z": -1.0,
      "ang_vel_xy": -0.05,
      "orientation": -2.0,  # gravity not straight down in the body frame
      "base_height": -10.0,
      # posture / regularisation
      "pose": -0.5,  # weighted deviation from the home pose (per-joint weights below)
      "stand_still": -0.2,  # joint deviation when the command is zero and the base is still
      "joint_limits": -5.0,
      "joint_vel": -1e-3,
      "joint_acc": -2.5e-7,
      "torques": -1e-5,
      "energy": -1e-4,  # |torque * joint velocity|
      "action_rate": -0.05,
      # applied after the positive clip
      "termination": -50.0,
  }


def _default_pose_weights():
  # Per-joint weight of (q - q_home)^2 in the "pose" term, by joint-name substring.
  # Low = free to move (walking joints, arm swing); high = keep near home.
  return {
      "hip_pitch": 0.01, "knee": 0.01, "ankle_pitch": 0.05,
      "hip_roll": 1.0, "hip_yaw": 1.0, "ankle_roll": 0.5,
      "waist_yaw": 1.0, "waist_roll": 2.0, "waist_pitch": 2.0,
      "shoulder_pitch": 0.1, "shoulder_roll": 1.0, "shoulder_yaw": 0.5,
      "elbow": 0.1, "wrist": 1.0,
  }


@dataclasses.dataclass(frozen=True)
class RewardConfig:
  weights: dict = dataclasses.field(default_factory=_default_weights)
  pose_weights: dict = dataclasses.field(default_factory=_default_pose_weights)
  tracking_sigma: float = 0.25
  base_height_target: float = 0.74  # m, pelvis; the bent-knee pose stands at ~0.74-0.75
  swing_height: float = 0.08  # m, peak foot clearance commanded by the gait clock
  feet_phase_sigma: float = 0.005
  air_time_target: float = 0.3  # s; touchdowns after shorter swings are penalised
  air_time_max_bonus: float = 0.2  # s; cap so the policy cannot farm long hops
  # Zero command but base faster than this (m/s, horizontal): the feet follow the
  # gait clock and stand_still is off, so a pushed robot may step to recover (v0.7).
  stand_recovery_speed: float = 0.25
  min_feet_distance: float = 0.15  # m, lateral; feet stand 0.24 apart at home
  soft_joint_limit: float = 0.95  # fraction of the joint range considered safe
  only_positive: bool = True  # clip the per-step sum at 0 (before termination)


@dataclasses.dataclass(frozen=True)
class DomainRandConfig:
  """Per-episode physics randomisation, sampled at every reset.

  Every range is scaled toward its nominal value by `level` in [0, 1]
  (0 = the nominal robot, 1 = the full ranges below). The level lives in the env
  state (info["dr_level"]) so the trainer can raise it as a curriculum.
  """
  level: float = 1.0  # initial level for reset()
  friction: tuple = (0.4, 1.0)  # absolute sliding friction of every floor contact (nominal 1.0)
  link_mass_scale: tuple = (0.9, 1.1)  # per body, inertia scaled with it
  torso_payload: tuple = (-1.0, 3.0)  # kg added to torso_link (nominal 33 kg robot)
  torso_com_offset: float = 0.03  # m, uniform per axis
  kp_scale: tuple = (0.9, 1.1)  # per actuator
  kd_scale: tuple = (0.8, 1.2)  # per actuator
  frictionloss_scale: tuple = (0.5, 2.0)  # per joint (nominal 0.1 Nm)
  armature_scale: tuple = (0.95, 1.05)  # per joint (motor reflected inertia)
  latency_prob: float = 0.5  # chance an episode applies each action one control step late
  push_interval: tuple = (5.0, 10.0)  # s between pushes
  push_velocity: tuple = (0.1, 0.6)  # m/s, horizontal base velocity kick, random direction


@dataclasses.dataclass(frozen=True)
class EnvConfig:
  sim: SimConfig = SimConfig()
  noise: NoiseConfig = NoiseConfig()
  command: CommandConfig = CommandConfig()
  reward: RewardConfig = RewardConfig()
  dr: DomainRandConfig = DomainRandConfig()
  episode_length: int = 1000  # control steps (20 s)
  action_scale: float = 0.5  # rad; target = home + scale * action, action in [-1, 1]
  gait_freq: tuple = (1.25, 1.5)  # Hz, sampled per episode
  # Reset distribution
  init_joint_noise: float = 0.05  # rad, uniform
  init_base_vel_noise: float = 0.1  # m/s and rad/s, uniform
  init_xy_range: float = 0.5  # m, uniform spawn offset
  init_yaw_range: float = 3.141592653589793  # rad, uniform heading (evaluation uses 0)
  # Termination
  min_base_height: float = 0.5  # pelvis z (m); stands at ~0.75
  max_tilt_cos: float = 0.5  # terminate if body-up · world-up < this (tilt > 60 deg)
  contact_threshold: float = 1e-3  # foot-floor signed distance (m) counted as contact



@dataclasses.dataclass(frozen=True)
class PPOConfig:
  # data per iteration = num_envs * unroll_length (8192 * 24 = 196,608 env steps)
  num_envs: int = 8192
  unroll_length: int = 24
  total_env_steps: float = 250e6
  # optimisation (rsl_rl-style; reference values used for legged robots)
  num_epochs: int = 5
  num_minibatches: int = 4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_eps: float = 0.2
  value_clip_eps: float = 0.2
  value_coef: float = 1.0
  entropy_coef: float = 0.005
  max_grad_norm: float = 1.0
  learning_rate: float = 1e-3  # initial; adapted per minibatch to keep KL near desired_kl
  adaptive_lr: bool = True
  desired_kl: float = 0.01
  lr_bounds: tuple = (1e-5, 1e-2)
  # networks
  policy_hidden: tuple = (512, 256, 128)
  value_hidden: tuple = (512, 256, 128)
  activation: str = "elu"
  init_std: float = 0.6
  obs_clip: float = 10.0  # after normalisation
  # domain-randomisation curriculum (level 0 = nominal robot, 1 = full ranges)
  dr_level_start: float = 0.0
  dr_level_step: float = 0.02  # added after each iteration that meets both criteria below
  curriculum_tracking: float = 0.6  # tracking_lin_vel reward >= this fraction of its max
  curriculum_survival_steps: float = 800.0  # falls rarer than once per this many env steps
  # bookkeeping
  seed: int = 0
  log_every: int = 10  # console line every N iterations (metrics file gets every iteration)
  ckpt_every: int = 50
  keep_ckpts: int = 3

  @property
  def batch_size(self) -> int:
    return self.num_envs * self.unroll_length

  @property
  def num_iterations(self) -> int:
    return int(-(-self.total_env_steps // self.batch_size))


PRESETS = {
    # Full overnight run on Kaggle 2xT4 (~5.5-6 h for 250M steps).
    "full": PPOConfig(),
    # < 5 min check: tiny batch, a few iterations, frequent logs/checkpoints.
    "smoke": PPOConfig(num_envs=1024, unroll_length=16, total_env_steps=10 * 1024 * 16,
                       log_every=1, ckpt_every=5, dr_level_start=0.5),
}
