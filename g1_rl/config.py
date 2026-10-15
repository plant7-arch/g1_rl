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
