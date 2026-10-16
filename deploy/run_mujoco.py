"""Standalone runner for the exported G1 walking policy (policy.onnx).

Needs only:  pip install numpy mujoco onnxruntime      (onnxruntime-gpu for CUDA)
No g1_rl, no JAX. Everything the controller needs (observation layout, joint order,
home pose, PD gains, action scale) is read from the ONNX file's metadata, so this
file doubles as the reference for writing a controller for the real robot.

  python run_mujoco.py --xml path/to/unitree_g1/scene.xml --viewer           # keyboard
  python run_mujoco.py --xml path/to/unitree_g1/scene.xml --cmd 0.5 0 0 --seconds 10

Viewer keys: Up/Down forward speed, Left/Right turn rate, , / . sideways speed,
Space stop, P push sideways, R reset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import mujoco
import numpy as np
import onnxruntime as ort


def quat_rotate_inverse(q, v):
  w, xyz = q[0], q[1:]
  t = 2.0 * np.cross(xyz, v)
  return v - w * t + np.cross(xyz, t)


class Controller:
  """Policy + PD-target mapping, exactly as during training (50 Hz)."""

  def __init__(self, onnx_path: str):
    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    self.session = ort.InferenceSession(onnx_path, providers=providers)
    self.meta = json.loads(self.session.get_modelmeta().custom_metadata_map["g1_rl"])
    m = self.meta
    self.layout = {k: slice(a, b) for k, (a, b) in m["obs_layout"].items()}
    self.home = np.asarray(m["home_ctrl"])
    self.kp, self.kd = np.asarray(m["kp"]), np.asarray(m["kd"])
    self.dt = float(m["ctrl_dt"])
    self.gait_freq = float(np.mean(m.get("gait_freq_range", [1.25, 1.5])))
    print(f"policy: iteration {m.get('iteration')}, running on {self.session.get_providers()[0]}")
    self.reset()

  def reset(self):
    self.last_action = np.zeros(len(self.home))
    self.phase = np.array([0.0, math.pi])

  def __call__(self, gyro, gravity, command, joint_pos, joint_vel):
    """All inputs in the policy's joint order. Returns PD position targets (rad)."""
    L = self.layout
    obs = np.zeros(self.meta["obs_size"], np.float32)
    obs[L["gyro"]], obs[L["gravity"]], obs[L["command"]] = gyro, gravity, command
    obs[L["joint_pos"]] = joint_pos - self.home
    obs[L["joint_vel"]] = joint_vel
    obs[L["last_act"]] = self.last_action
    obs[L["gait"]] = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
    action, targets = self.session.run(None, {"obs": obs[None]})
    self.last_action = action[0].astype(np.float64)
    self.phase = (self.phase + 2 * math.pi * self.gait_freq * self.dt + math.pi) % (2 * math.pi) - math.pi
    return targets[0].astype(np.float64)


class Sim:
  """The robot in MuJoCo, with the policy's PD gains on its position actuators."""

  def __init__(self, xml: str, ctrl: Controller):
    self.m = mujoco.MjModel.from_xml_path(xml)
    self.d = mujoco.MjData(self.m)
    self.ctrl = ctrl
    names = ctrl.meta["joint_names"]
    jid = [self.m.joint(n).id for n in names]
    self.qadr, self.vadr = self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]
    by_joint = {int(self.m.actuator_trnid[a, 0]): a for a in range(self.m.nu)}
    self.act = np.array([by_joint[j] for j in jid])
    self.m.actuator_gainprm[self.act, 0] = ctrl.kp
    self.m.actuator_biasprm[self.act, 1] = -ctrl.kp
    self.m.actuator_biasprm[self.act, 2] = -ctrl.kd
    self.n_sub = max(1, round(ctrl.dt / self.m.opt.timestep))
    self.command = np.zeros(3)
    self.reset()

  def reset(self):
    m, d = self.m, self.d
    mujoco.mj_resetData(m, d)
    d.qpos[2], d.qpos[3:7] = 0.8, (1, 0, 0, 0)
    d.qpos[self.qadr] = self.ctrl.home
    d.ctrl[self.act] = self.ctrl.home
    mujoco.mj_forward(m, d)
    d.qpos[2] -= self._foot_clearance()
    mujoco.mj_forward(m, d)
    self.ctrl.reset()

  def _foot_clearance(self):
    feet = {self.m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")}
    m, f = self.m, self.m.geom("floor").id

    def touches_floor(g):  # explicit contact pair (scene_mjx.xml) or contype/conaffinity (scene.xml)
      paired = m.npair and np.any(((m.pair_geom1 == g) & (m.pair_geom2 == f)) |
                                  ((m.pair_geom1 == f) & (m.pair_geom2 == g)))
      return bool(paired or (m.geom_contype[g] & m.geom_conaffinity[f]) or
                  (m.geom_contype[f] & m.geom_conaffinity[g]))

    return min(mujoco.mj_geomDistance(m, self.d, g, f, 1.0, None)
               for g in range(m.ngeom) if m.geom_bodyid[g] in feet and touches_floor(g))

  def step(self, push=None):
    d = self.d
    q = d.qpos[3:7]
    targets = self.ctrl(gyro=d.qvel[3:6], gravity=quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0])),
                        command=self.command, joint_pos=d.qpos[self.qadr], joint_vel=d.qvel[self.vadr])
    d.ctrl[self.act] = targets
    if push is not None:
      d.qvel[0:2] += push
    for _ in range(self.n_sub):
      mujoco.mj_step(self.m, d)

  def fallen(self):
    q = self.d.qpos
    w, x, y, _ = q[3:7]
    up_z = 1 - 2 * (x * x + y * y)
    return bool(q[2] < 0.5 or up_z < 0.5)


def main():
  here = os.path.dirname(os.path.abspath(__file__))
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--onnx", default=os.path.join(here, "policy.onnx"))
  p.add_argument("--xml", default=os.path.join(here, "unitree_g1", "scene.xml"),
                 help="MuJoCo Menagerie unitree_g1 scene.xml (or scene_mjx.xml)")
  p.add_argument("--cmd", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
  p.add_argument("--seconds", type=float, default=10.0)
  p.add_argument("--viewer", action="store_true", help="interactive viewer with keyboard commands")
  args = p.parse_args()

  sim = Sim(args.xml, Controller(args.onnx))
  if not args.viewer:
    sim.command = np.asarray(args.cmd)
    steps = int(args.seconds / sim.ctrl.dt)
    xy0 = sim.d.qpos[0:2].copy()
    t0 = time.time()
    for t in range(steps):
      sim.step()
      if sim.fallen():
        print(f"fell after {t * sim.ctrl.dt:.2f} s")
        return 1
    dist = np.linalg.norm(sim.d.qpos[0:2] - xy0)
    print(f"walked {dist:.2f} m in {args.seconds:.0f} s with command {tuple(args.cmd)} "
          f"(simulated {args.seconds / (time.time() - t0):.1f}x real time), no fall")
    return 0

  import mujoco.viewer  # noqa: PLC0415
  keys = {265: ("vx", 0.1), 264: ("vx", -0.1), 263: ("wz", 0.1), 262: ("wz", -0.1),
          44: ("vy", 0.1), 46: ("vy", -0.1)}
  limits = {"vx": (-0.6, 1.0), "vy": (-0.5, 0.5), "wz": (-1.0, 1.0)}
  idx = {"vx": 0, "vy": 1, "wz": 2}
  ev = {"push": False, "reset": False}

  def on_key(k):
    if k in keys:
      name, dv = keys[k]
      lo, hi = limits[name]
      sim.command[idx[name]] = round(min(max(sim.command[idx[name]] + dv, lo), hi), 2)
    elif k == 32:
      sim.command[:] = 0
    elif k == 80:
      ev["push"] = True
    elif k == 82:
      ev["reset"] = True
    else:
      return
    print("command vx {:+.1f} vy {:+.1f} wz {:+.1f}".format(*sim.command), flush=True)

  with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=on_key) as v:
    v.cam.type, v.cam.trackbodyid, v.cam.distance = mujoco.mjtCamera.mjCAMERA_TRACKING, sim.m.body("pelvis").id, 3.0
    while v.is_running():
      t0 = time.time()
      if ev["reset"] or sim.fallen():
        sim.reset()
        ev["reset"] = False
      push = np.array([0.0, 0.5]) if ev["push"] else None
      ev["push"] = False
      sim.step(push)
      v.sync()
      time.sleep(max(0.0, sim.ctrl.dt - (time.time() - t0)))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
