"""Render a saved trajectory (.npz with qpos, fps, xml_path) to mp4.

  python scripts/render_traj.py outputs/smoke/traj.npz [--out video.mp4]

On a desktop (not headless) unset MUJOCO_GL or use MUJOCO_GL=glfw.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

from g1_rl import render  # noqa: E402
from g1_rl.config import SimConfig  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("npz")
  p.add_argument("--out", default=None)
  p.add_argument("--xml", default=None, help="override model path (default: from npz, else training scene)")
  a = p.parse_args()
  xml = a.xml
  if xml is None:
    import numpy as np
    stored = str(np.load(a.npz)["xml_path"])
    xml = stored if os.path.exists(stored) else SimConfig().xml_path
  out = a.out or os.path.splitext(a.npz)[0] + ".mp4"
  path = render.try_render(a.npz, out, xml_path=xml)
  print(f"wrote {path}" if path else "rendering failed")


if __name__ == "__main__":
  main()
