"""Drive a trained policy interactively in the MuJoCo viewer (on your own computer).

  python scripts/play.py --policy outputs/runs/g1_walk_v1/policy_final.npz
  python scripts/play.py --policy ... --model mjx      # simplified training model instead of full meshes
  python scripts/play.py --policy outputs/export/g1_walk_v1/policy.onnx   # exported ONNX (onnxruntime)

Keys (focus the viewer window):
  Up / Down      forward speed +/- 0.1 m/s        Left / Right   turn rate +/- 0.1 rad/s
  , / .          sideways speed +/- 0.1 m/s       Space          stop (all commands 0)
  P              push the robot sideways (0.5 m/s) R              reset

Runs in real time; the policy runs on the GPU if available (JAX for .npz, onnxruntime
for .onnx), else on the CPU.
Needs a display (not for Kaggle). Uses the same runner as the sim-to-sim evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE, KEY_COMMA, KEY_PERIOD, KEY_P, KEY_R = 32, 44, 46, 80, 82
LIMITS = {"vx": (-0.6, 1.0), "vy": (-0.5, 0.5), "wz": (-1.0, 1.0)}  # training command ranges


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--policy", required=True)
  p.add_argument("--model", default="full", choices=["full", "mjx"])
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  args = p.parse_args()
  os.environ.pop("MUJOCO_GL", None)  # use the desktop OpenGL for the viewer

  from g1_rl import device
  rt = device.setup(args.platform)
  os.environ.pop("MUJOCO_GL", None)
  import mujoco.viewer
  import numpy as np

  from g1_rl.config import SimConfig
  from g1_rl.export import load_any
  from g1_rl.sim2sim import MujocoRunner

  pi = load_any(args.policy, device=rt.devices[0])
  xml = SimConfig().eval_xml_path if args.model == "full" else SimConfig().xml_path
  runner = MujocoRunner(xml, pi)
  cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
  events = {"push": False, "reset": False}

  def clamp(k, v):
    lo, hi = LIMITS[k]
    cmd[k] = round(min(max(v, lo), hi), 2)

  def on_key(key):
    if key == KEY_UP: clamp("vx", cmd["vx"] + 0.1)
    elif key == KEY_DOWN: clamp("vx", cmd["vx"] - 0.1)
    elif key == KEY_LEFT: clamp("wz", cmd["wz"] + 0.1)
    elif key == KEY_RIGHT: clamp("wz", cmd["wz"] - 0.1)
    elif key == KEY_COMMA: clamp("vy", cmd["vy"] + 0.1)
    elif key == KEY_PERIOD: clamp("vy", cmd["vy"] - 0.1)
    elif key == KEY_SPACE: cmd.update(vx=0.0, vy=0.0, wz=0.0)
    elif key == KEY_P: events["push"] = True
    elif key == KEY_R: events["reset"] = True
    else: return
    print(f"command vx {cmd['vx']:+.1f} m/s  vy {cmd['vy']:+.1f} m/s  wz {cmd['wz']:+.1f} rad/s", flush=True)

  runner.reset()
  print(__doc__.split("Keys")[1].split("Runs in")[0])
  with mujoco.viewer.launch_passive(runner.m, runner.d, key_callback=on_key) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = runner.m.body("pelvis").id
    viewer.cam.distance = 3.0
    while viewer.is_running():
      t0 = time.time()
      if events["reset"] or runner.fallen():
        if runner.fallen():
          print("fell -> reset")
        runner.reset()
        events["reset"] = False
      runner.command = np.array([cmd["vx"], cmd["vy"], cmd["wz"]])
      push = None
      if events["push"]:
        push, events["push"] = np.array([0.0, 0.5]), False
      runner.step(push)
      viewer.sync()
      time.sleep(max(0.0, runner.ctrl_dt - (time.time() - t0)))


if __name__ == "__main__":
  main()
