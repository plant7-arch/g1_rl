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


@dataclasses.dataclass(frozen=True)
class EnvConfig:
  sim: SimConfig = SimConfig()
  noise: NoiseConfig = NoiseConfig()
  command: CommandConfig = CommandConfig()
  episode_length: int = 1000  # control steps (20 s)
  action_scale: float = 0.5  # rad; target = home + scale * action, action in [-1, 1]
  gait_freq: tuple = (1.25, 1.5)  # Hz, sampled per episode
  # Reset distribution
  init_joint_noise: float = 0.05  # rad, uniform
  init_base_vel_noise: float = 0.1  # m/s and rad/s, uniform
  init_xy_range: float = 0.5  # m, uniform spawn offset
  # Termination
  min_base_height: float = 0.5  # pelvis z (m); stands at ~0.75
  max_tilt_cos: float = 0.5  # terminate if body-up · world-up < this (tilt > 60 deg)
  contact_threshold: float = 1e-3  # foot-floor signed distance (m) counted as contact

