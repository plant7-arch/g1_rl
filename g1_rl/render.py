"""Offscreen video rendering of qpos trajectories with classic MuJoCo.

Headless backends: MUJOCO_GL=egl (NVIDIA GPU, e.g. Kaggle T4) or osmesa (CPU).
If no backend works, callers still have the saved .npz trajectory and can
render it later with `python scripts/render_traj.py traj.npz` on any machine
with a display or GPU.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np


def save_trajectory(path: str, qpos: np.ndarray, fps: float, xml_path: str, **extra) -> str:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  np.savez_compressed(path, qpos=np.asarray(qpos, np.float32), fps=fps, xml_path=xml_path, **extra)
  return path


def render_qpos(xml_path: str, qpos: np.ndarray, width: int = 640, height: int = 480,
                track_body: str = "pelvis", distance: float = 3.0, azimuth: float = 135.0,
                elevation: float = -15.0) -> list:
  m = mujoco.MjModel.from_xml_path(xml_path)
  width = min(width, m.vis.global_.offwidth)
  height = min(height, m.vis.global_.offheight)
  d = mujoco.MjData(m)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = m.body(track_body).id
  cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
  frames = []
  with mujoco.Renderer(m, height, width) as r:
    for q in qpos:
      d.qpos[:] = q
      mujoco.mj_forward(m, d)
      r.update_scene(d, camera=cam)
      frames.append(r.render().copy())
  return frames


def write_video(path: str, frames: list, fps: float) -> str:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  try:
    import mediapy  # noqa: PLC0415
    mediapy.write_video(path, frames, fps=fps)
  except ImportError:
    import imageio  # noqa: PLC0415
    imageio.mimsave(path, frames, fps=fps)
  return path


def try_render(npz_or_qpos, out_mp4: str, xml_path: str | None = None, fps: float | None = None,
               **kw) -> str | None:
  """Render to mp4; returns the path, or None (with a printed reason) if rendering fails."""
  if isinstance(npz_or_qpos, str):
    z = np.load(npz_or_qpos)
    qpos, fps, xml_path = z["qpos"], float(z["fps"]), xml_path or str(z["xml_path"])
  else:
    qpos = npz_or_qpos
  try:
    return write_video(out_mp4, render_qpos(xml_path, qpos, **kw), fps or 50)
  except Exception as e:  # no GL backend, missing ffmpeg, ...
    print(f"[render] skipped ({type(e).__name__}: {str(e)[:120]}). "
          f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}; trajectory is still saved.")
    return None
