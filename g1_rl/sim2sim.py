"""Sim-to-sim: run an exported policy in classic (CPU, float64) MuJoCo.

This runner deliberately does NOT use the training environment code. It builds
observations and maps actions only from what is stored in the policy file
(joint names, home pose, observation layout, action scale, PD gains), exactly
as a deployment controller would. If the policy also works here, and in the
full-mesh model (assets/scene.xml), it does not depend on MJX quirks.

Differences of the full model vs training are reported, not hidden:
mesh collisions, 2 ms timestep, 100 solver iterations, joint friction 0.3 Nm
(3x training nominal). Its stiff default PD gains (kp 500) are replaced by the
policy's gains, since those are what you would set on the robot.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def _quat_rotate_inverse(q, v):
  w, xyz = q[0], q[1:]
  t = 2.0 * np.cross(xyz, v)
  return v - w * t + np.cross(xyz, t)


def _quat_rotate(q, v):
  w, xyz = q[0], q[1:]
  t = 2.0 * np.cross(xyz, v)
  return v + w * t + np.cross(xyz, t)


def _yaw(q):
  w, x, y, z = q
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MujocoRunner:

  def __init__(self, xml_path: str, policy, use_policy_gains: bool = True,
               gait_freq: float | None = None, overrides: dict | None = None):
    self.m = mujoco.MjModel.from_xml_path(xml_path)
    self.d = mujoco.MjData(self.m)
    self.policy = policy
    meta = policy.meta
    self.layout = {k: slice(*v) for k, v in meta["obs_layout"].items()}
    self.obs_size = meta["obs_size"]
    self.history = int(meta.get("history", 1))  # frames stacked newest first (v0.9)
    self.frame_size = int(meta.get("frame_size", self.obs_size))
    self.ctrl_dt = float(meta["ctrl_dt"])
    self.n_sub = max(1, round(self.ctrl_dt / self.m.opt.timestep))
    if abs(self.n_sub * self.m.opt.timestep - self.ctrl_dt) > 1e-9:
      raise ValueError(f"control dt {self.ctrl_dt} is not a multiple of the model timestep "
                       f"{self.m.opt.timestep}")
    lo_hi = meta.get("gait_freq_range", [1.25, 1.5])
    self.gait_freq = gait_freq or 0.5 * (lo_hi[0] + lo_hi[1])

    # Map the policy's joint order onto this model by joint name.
    names = meta["joint_names"]
    jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if min(jid) < 0:
      raise KeyError(f"model lacks joints: {[n for n, j in zip(names, jid) if j < 0]}")
    self.qadr = self.m.jnt_qposadr[jid]
    self.vadr = self.m.jnt_dofadr[jid]
    act_by_joint = {int(self.m.actuator_trnid[a, 0]): a for a in range(self.m.nu)}
    self.act = np.array([act_by_joint[j] for j in jid])
    if use_policy_gains:
      self.m.actuator_gainprm[self.act, 0] = policy.kp
      self.m.actuator_biasprm[self.act, 1] = -policy.kp
      self.m.actuator_biasprm[self.act, 2] = -policy.kd
    for field, value in (overrides or {}).items():  # e.g. {"opt.timestep": 0.002}
      obj, attr = field.split(".") if "." in field else (None, field)
      setattr(getattr(self.m, obj) if obj else self.m, attr, value)
    self.mass = float(self.m.body_subtreemass[self.m.body("pelvis").id])
    self.foot_bodies = [self.m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self.reset()

  # ------------------------------------------------------------------ episode
  def reset(self, command=(0.0, 0.0, 0.0), yaw: float = 0.0):
    m, d = self.m, self.d
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.8)
    d.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    d.qpos[self.qadr] = self.policy.home
    d.ctrl[self.act] = self.policy.home
    mujoco.mj_forward(m, d)
    d.qpos[2] -= self._lowest_foot_point()
    mujoco.mj_forward(m, d)
    self.command = np.asarray(command, np.float64)
    self.last_action = np.zeros(len(self.qadr))
    self.phase = np.array([0.0, math.pi])
    self.obs_hist = None  # last stacked observation (None = new episode: fill with the frame)
    self.t = 0

  def _lowest_foot_point(self) -> float:
    """Signed distance of the feet's collision geoms to the floor (works for meshes)."""
    best = math.inf
    for g in range(self.m.ngeom):
      if self.m.geom_bodyid[g] in self.foot_bodies and self._collides_with_floor(g):
        dist = mujoco.mj_geomDistance(self.m, self.d, g, self.floor, 1.0, None)
        best = min(best, dist)
    if not math.isfinite(best):
      raise RuntimeError("no colliding foot geoms found")
    return best

  def _collides_with_floor(self, g) -> bool:
    m, f = self.m, self.floor
    if m.npair and np.any(((m.pair_geom1 == g) & (m.pair_geom2 == f)) | ((m.pair_geom1 == f) & (m.pair_geom2 == g))):
      return True
    return bool((m.geom_contype[g] & m.geom_conaffinity[f]) or (m.geom_contype[f] & m.geom_conaffinity[g]))

  def observation(self) -> np.ndarray:
    """The policy input for this control step: the current frame + the previous ones.
    Call once per control step (it advances the history)."""
    f = self.frame()
    if self.obs_hist is None:
      obs = np.tile(f, self.history)
    else:
      obs = np.concatenate([f, self.obs_hist[:-self.frame_size]]) if self.history > 1 else f
    self.obs_hist = obs
    return obs

  def frame(self) -> np.ndarray:
    """One observation frame from the current state (same layout as the training env)."""
    d, L = self.d, self.layout
    q = d.qpos[3:7]
    obs = np.zeros(self.frame_size, np.float32)
    obs[L["gyro"]] = d.qvel[3:6]  # free-joint angular velocity is in the body frame
    obs[L["gravity"]] = _quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
    obs[L["command"]] = self.command
    obs[L["joint_pos"]] = d.qpos[self.qadr] - self.policy.home
    obs[L["joint_vel"]] = d.qvel[self.vadr]
    obs[L["last_act"]] = self.last_action
    obs[L["gait"]] = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
    return obs

  def step(self, push_vel=None):
    obs = self.observation()
    a = np.asarray(self.policy(obs), np.float64)
    targets = np.clip(self.policy.home + self.policy.action_scale * a, self.policy.lo, self.policy.hi)
    self.d.ctrl[self.act] = targets
    if push_vel is not None:
      self.d.qvel[0:2] += push_vel
    for _ in range(self.n_sub):
      mujoco.mj_step(self.m, self.d)
    self.last_action = a
    self.phase = (self.phase + 2 * math.pi * self.gait_freq * self.ctrl_dt + math.pi) % (2 * math.pi) - math.pi
    self.t += 1
    return a

  def fallen(self) -> bool:
    q = self.d.qpos
    up_z = _quat_rotate(q[3:7], np.array([0.0, 0.0, 1.0]))[2]
    return bool(q[2] < 0.5 or up_z < 0.5 or not np.all(np.isfinite(q)))

  def heading_velocity(self):
    q = self.d.qpos[3:7]
    yaw = _yaw(q)
    hq = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    return _quat_rotate_inverse(hq, self.d.qvel[0:3])[:2], float(self.d.qvel[5])

  # ------------------------------------------------------------------ scenario
  def run(self, command, seconds: float = 10.0, push_time: float | None = None,
          push_vel=(0.0, 0.0), settle_s: float = 2.0, record_qpos: bool = False) -> dict:
    self.reset(command)
    steps = int(round(seconds / self.ctrl_dt))
    push_step = None if push_time is None else int(round(push_time / self.ctrl_dt))
    v_err, w_err, vx, power, qpos = [], [], [], [], []
    fell_at = None
    for t in range(steps):
      self.step(np.asarray(push_vel) if t == push_step else None)
      if record_qpos:
        qpos.append(self.d.qpos.copy())
      if self.fallen():
        fell_at = t
        break
      v, wz = self.heading_velocity()
      in_window = t >= settle_s / self.ctrl_dt and (push_step is None or t < push_step)
      if in_window:
        v_err.append(np.linalg.norm(v - self.command[:2]))
        w_err.append(abs(wz - self.command[2]))
        vx.append(v[0])
        jv = self.d.qvel[self.vadr]
        power.append(np.sum(np.abs(self.d.actuator_force[self.act] * jv)))
    out = {"survived": fell_at is None,
           "time_alive_s": (fell_at if fell_at is not None else steps) * self.ctrl_dt,
           "lin_vel_err": float(np.mean(v_err)) if v_err else None,
           "yaw_rate_err": float(np.mean(w_err)) if w_err else None,
           "vx_mean": float(np.mean(vx)) if vx else None}
    speed = abs(np.mean(vx)) if vx else 0.0
    out["cost_of_transport"] = (float(np.mean(power) / (self.mass * 9.81 * speed))
                                if speed > 0.2 else None)
    if record_qpos:
      out["qpos"] = np.asarray(qpos)
    return out


SIM2SIM_SCENARIOS = [
    ("stand", (0.0, 0.0, 0.0), None, (0, 0)),
    ("forward 0.5", (0.5, 0.0, 0.0), None, (0, 0)),
    ("forward 1.0", (1.0, 0.0, 0.0), None, (0, 0)),
    ("backward 0.4", (-0.4, 0.0, 0.0), None, (0, 0)),
    ("sideways 0.3", (0.0, 0.3, 0.0), None, (0, 0)),
    ("turn 0.8", (0.0, 0.0, 0.8), None, (0, 0)),
    ("walk+turn", (0.5, 0.0, 0.5), None, (0, 0)),
    ("walk, push left 0.5", (0.5, 0.0, 0.0), 4.0, (0.0, 0.5)),
    ("stand, push front 0.5", (0.0, 0.0, 0.0), 4.0, (0.5, 0.0)),
]


def run_sim2sim(policy, models: dict, seconds: float = 10.0, video_dir: str | None = None,
                video_scenarios=("forward 0.5", "walk, push left 0.5"), video_models=None) -> dict:
  """models: {label: xml_path}. Returns {label: {scenario: metrics}}."""
  from g1_rl import render  # noqa: PLC0415

  results = {}
  for label, xml in models.items():
    runner = MujocoRunner(xml, policy)
    res = {"_model": {"timestep": runner.m.opt.timestep, "solver_iterations": int(runner.m.opt.iterations),
                      "substeps": runner.n_sub,
                      "joint_frictionloss": float(np.median(runner.m.dof_frictionloss[runner.vadr])),
                      "contacts": "explicit pairs" if runner.m.npair else "all geoms (meshes)"}}
    for name, cmd, push_t, push_v in SIM2SIM_SCENARIOS:
      want_video = (video_dir is not None and name in video_scenarios
                    and (video_models is None or label in video_models))
      r = runner.run(cmd, seconds, push_t, push_v, record_qpos=want_video)
      if want_video:
        slug = f"{label}_{name}".replace(" ", "_").replace(",", "").replace(".", "p")
        traj = render.save_trajectory(f"{video_dir}/{slug}.npz", r.pop("qpos"),
                                      fps=1 / runner.ctrl_dt, xml_path=xml)
        r["video"] = render.try_render(traj, f"{video_dir}/{slug}.mp4")
      res[name] = r
    results[label] = res
  return results
