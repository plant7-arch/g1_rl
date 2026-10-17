"""Evaluation suite in MJX (GPU by default, CPU fallback).

Every scenario is one environment in a single batch, so the whole suite is one
compile and one rollout. Scenario parameters (command, physics, pushes) are
written into the per-env state after reset; the deterministic policy (mean
action of the exported policy file) is evaluated on all of them at once.

Groups
  grid       nominal robot, fixed velocity commands over a grid (tracking accuracy)
  training   full randomisation, random commands and pushes as in training (robustness)
  ood        one physics parameter pushed beyond the training range (generalisation)
  push       a single push of increasing strength in 4 directions (push recovery)
  slope      standing / walking on 3-9 deg slopes, up, down and across (v0.9)
  imu        IMU mounted 3 deg off in pitch or roll (v0.9)
  small      small commands (0.15-0.2 m/s, 0.2-0.3 rad/s): does it move at all? (v0.9)

Per-step signals are recorded on device; metrics are computed with NumPy afterwards.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from g1_rl import math as gm
from g1_rl import parallel
from g1_rl.config import EnvConfig, NoiseConfig
from g1_rl.env import G1Env
from g1_rl.vec_env import VecEnv

NEVER = 10 ** 7


@dataclasses.dataclass
class Scenario:
  group: str
  name: str
  command: tuple = (0.0, 0.0, 0.0)
  dr_level: float = 0.0  # 0 = nominal robot; 1 = training randomisation
  resample_commands: bool = False
  dr_override: dict = dataclasses.field(default_factory=dict)  # e.g. {"friction": 0.3}
  push_time: float | None = None  # s; None = no push (training group keeps random pushes)
  push_vel: tuple = (0.0, 0.0)  # m/s, world frame (robot starts facing +x)
  random_pushes: bool = False
  tag: dict = dataclasses.field(default_factory=dict)  # values for grouping in the report


def build_suite(quick: bool = False) -> list:
  rep = 1 if quick else 2
  s = []
  vxs = (-0.5, 0.0, 0.5, 1.0) if quick else (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
  vys = (0.0,) if quick else (-0.4, 0.0, 0.4)
  wzs = (-0.8, 0.0, 0.8)
  for vx in vxs:
    for vy in vys:
      for wz in wzs:
        for r in range(rep):
          s.append(Scenario("grid", f"cmd({vx:+.2f},{vy:+.2f},{wz:+.2f})", (vx, vy, wz),
                            tag={"vx": vx, "vy": vy, "wz": wz}))
  for i in range(32 if quick else 128):
    s.append(Scenario("training", f"train#{i}", dr_level=1.0, resample_commands=True,
                      random_pushes=True))
  ood = {
      "friction": (0.15, 0.2, 0.3, 0.6, 1.0, 1.5),
      "payload": (-2.0, 0.0, 4.0, 8.0, 12.0),
      "kp_scale": (0.6, 0.7, 0.8, 1.0, 1.3, 1.5),
      "kd_scale": (0.5, 1.0, 1.5, 2.0),
      "com_offset_x": (-0.06, -0.03, 0.0, 0.03, 0.06),
      "frictionloss_scale": (0.5, 1.0, 2.0, 3.0, 4.0),
      "latency": (0, 1, 2, 3),
  }
  if quick:
    ood = {k: (v[0], v[len(v) // 2], v[-1]) if len(v) > 2 else v for k, v in ood.items()}
  for param, values in ood.items():
    for v in values:
      for r in range(1 if quick else 4):
        s.append(Scenario("ood", f"{param}={v}", (0.5, 0.0, 0.0), dr_override={param: v},
                          tag={"param": param, "value": v}))
  mags = (0.5, 1.0, 1.5) if quick else (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
  dirs = {"front": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1)}
  for cmd_name, cmd in (("stand", (0.0, 0.0, 0.0)), ("walk", (0.5, 0.0, 0.0))):
    for dname, d in dirs.items():
      for m in mags:
        for r in range(rep):
          s.append(Scenario("push", f"{cmd_name}/{dname}/{m}", cmd, push_time=4.0,
                            push_vel=(m * d[0], m * d[1]),
                            tag={"cmd": cmd_name, "dir": dname, "mag": m}))
  # Slopes: gravity tilted by `deg`, pulling toward `downhill` (deg, robot faces +x).
  downhill = {"up": 180.0, "down": 0.0, "across": 90.0}
  for deg in ((3, 6) if quick else (3, 6, 9)):
    for dname, phi in downhill.items():
      for cmd_name, cmd in (("stand", (0.0, 0.0, 0.0)), ("walk", (0.5, 0.0, 0.0))):
        s.append(Scenario("slope", f"slope {deg} {dname}/{cmd_name}", cmd,
                          dr_override={"slope": (deg, phi)},
                          tag={"deg": deg, "dir": dname, "cmd": cmd_name}))
  for axis, sign in (("pitch", 1), ("pitch", -1), ("roll", 1)):
    for cmd_name, cmd in (("stand", (0.0, 0.0, 0.0)), ("walk", (0.5, 0.0, 0.0))):
      tilt = (0.0, sign * 3.0) if axis == "pitch" else (3.0, 0.0)
      s.append(Scenario("imu", f"imu {axis} {sign * 3:+d}deg/{cmd_name}", cmd,
                        dr_override={"imu_tilt_deg": tilt},
                        tag={"axis": axis, "deg": sign * 3, "cmd": cmd_name}))
  smalls = [(0.15, 0, 0), (0.2, 0, 0), (-0.15, 0, 0), (-0.2, 0, 0), (0, 0.15, 0), (0, -0.15, 0),
            (0, 0, 0.2), (0, 0, -0.2), (0, 0, 0.3), (0, 0, -0.3)]
  for c in (smalls[::2] if quick else smalls):
    s.append(Scenario("small", f"small cmd{c}", c, tag={"vx": c[0], "vy": c[1], "wz": c[2]}))
  return s


def eval_env_config(base: EnvConfig = EnvConfig(), episode_steps: int = 1000,
                    obs_noise: bool = True) -> EnvConfig:
  """Training env with deterministic start (facing +x at the origin) and no time limit."""
  return dataclasses.replace(
      base, episode_length=episode_steps + 10, init_xy_range=0.0, init_yaw_range=0.0,
      noise=base.noise if obs_noise else NoiseConfig(level=0.0))


def _apply_scenarios(env: G1Env, state, scenarios, steps: int):
  """Write each scenario's command / physics / pushes into the batched env state."""
  info = jax.device_get(state.info)
  info = jax.tree.map(np.array, info)  # writable host copies
  dt = env.dt
  for i, sc in enumerate(scenarios):
    if not sc.resample_commands:
      info["command"][i] = sc.command
      info["cmd_timer"][i] = NEVER
    for k, v in sc.dr_override.items():
      if k == "com_offset_x":
        info["dr"]["com_offset"][i] = (v, 0.0, 0.0)
      elif k == "latency":
        info["dr"]["latency"][i] = int(v)
      elif k == "slope":  # (tilt deg, direction gravity pulls in deg)
        th, ph = math.radians(v[0]), math.radians(v[1])
        info["dr"]["gravity_dir"][i] = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                                        -math.cos(th))
      elif k == "imu_tilt_deg":  # (roll, pitch) mounting error in deg
        info["dr"]["imu_tilt"][i] = np.radians(v)
      else:  # scalar params (friction, payload) or per-joint scales set uniformly
        info["dr"][k][i] = v
    if sc.push_time is not None:
      info["push_timer"][i] = int(round(sc.push_time / dt))
      info["push_kick"][i] = sc.push_vel
    elif not sc.random_pushes:
      info["push_timer"][i] = NEVER
  return state._replace(info=info)


def run_mjx_suite(policy, rt, scenarios=None, seconds: float = 20.0, quick: bool = False,
                  obs_noise: bool = True, seed: int = 0, record_qpos_for=()):
  """Run all scenarios in one batch. Returns (per-scenario metrics list, extras)."""
  scenarios = scenarios if scenarios is not None else build_suite(quick)
  steps = int(round(seconds / 0.02))
  env = G1Env(eval_env_config(EnvConfig(), steps, obs_noise))
  n_dev = rt.n_devices
  n = len(scenarios)
  n_pad = -(-n // n_dev) * n_dev
  padded = scenarios + [scenarios[-1]] * (n_pad - n)

  venv = VecEnv(env, rt.mesh, n_pad)
  levels = parallel.put_batched(jnp.asarray([sc.dr_level for sc in padded], jnp.float32), rt.mesh)
  state = venv.reset_fn(venv.keys(seed), levels)
  state = parallel.put_batched(_apply_scenarios(env, state, padded, steps), rt.mesh)
  refresh = parallel.jit_batched(jax.vmap(env.refresh_obs), rt.mesh)
  state = refresh(state, venv.keys(seed + 2))  # first observation sees the scenario's command

  pi = policy.jax_fn()
  qadr = env._qadr
  feet_site = env._feet_site
  record_idx = np.asarray(list(record_qpos_for), np.int32)

  def record(s):
    d = env.packer.unpack(s.data)
    quat = d.qpos[3:7]
    heading = gm.yaw_quat(gm.quat_to_yaw(quat))
    v = gm.quat_rotate_inverse(heading, d.qvel[0:3])
    up_z = gm.quat_rotate(quat, jnp.array([0.0, 0.0, 1.0]))[2]
    jvel = d.qvel[env._vadr]
    return {
        "fell": s.metrics["episode/fell"],
        "v": v[:2], "wz": d.qvel[5], "cmd": s.info["command"],
        "z": d.qpos[2], "up_z": up_z,
        "power": jnp.sum(jnp.abs(d.actuator_force * jvel)),
        "torque_ratio": jnp.max(jnp.abs(d.actuator_force) / env._torque_limit),
        "contact": s.info["contact"].astype(jnp.float32),
        "foot_xy": d.site_xpos[feet_site][:, :2],
        "foot_h": env.foot_heights(d),
        "joint_margin": jnp.min(jnp.minimum(d.qpos[qadr] - env._lo, env._hi - d.qpos[qadr])),
        "qpos": d.qpos,
    }

  rollout = venv.make_rollout(lambda obs, key: pi(obs), steps, record)
  _, _, rec, _ = rollout(state, venv.keys(seed + 1))
  rec = jax.device_get(rec)
  rec = {k: np.asarray(v)[:n] for k, v in rec.items()}
  qpos_rec = {int(i): rec["qpos"][int(i)] for i in record_idx}
  del rec["qpos"]
  metrics = [analyse(sc, {k: v[i] for k, v in rec.items()}, env) for i, sc in enumerate(scenarios)]
  return metrics, {"qpos": qpos_rec, "env": env, "steps": steps}


def analyse(sc: Scenario, r: dict, env: G1Env, settle_s: float = 2.0) -> dict:
  """Metrics for one scenario from its recorded time series (arrays of length T)."""
  dt = env.dt
  T = len(r["fell"])
  fell_steps = np.flatnonzero(r["fell"] > 0)
  t_fall = int(fell_steps[0]) if len(fell_steps) else T
  alive = np.arange(T) < t_fall
  settle = int(settle_s / dt)
  win = alive & (np.arange(T) >= settle)
  if sc.push_time is not None:  # judge tracking before the push, survival after it
    win &= np.arange(T) < int(sc.push_time / dt)
  out = {"group": sc.group, "name": sc.name, **sc.tag,
         "survived": bool(t_fall == T), "time_alive_s": t_fall * dt}
  if win.sum() > 10:
    cmd, v, wz = r["cmd"][win], r["v"][win], r["wz"][win]
    out["lin_vel_err"] = float(np.mean(np.linalg.norm(v - cmd[:, :2], axis=1)))
    out["yaw_rate_err"] = float(np.mean(np.abs(wz - cmd[:, 2])))
    out["vx_mean"], out["vy_mean"], out["wz_mean"] = (float(np.mean(v[:, 0])), float(np.mean(v[:, 1])),
                                                      float(np.mean(wz)))
    speed = np.linalg.norm(v, axis=1).mean()
    mass = float(env.robot.total_mass)
    out["cost_of_transport"] = (float(np.mean(r["power"][win]) / (mass * 9.81 * speed))
                                if speed > 0.2 else None)
    c = r["contact"][win]
    fv = np.zeros((T, 2))  # horizontal foot speed from finite differences
    fv[1:] = np.linalg.norm(np.diff(r["foot_xy"], axis=0), axis=-1) / dt
    stance = c > 0.5
    out["foot_slip"] = float(fv[win][stance].mean()) if stance.any() else None
    touchdowns = ((c[1:] > 0.5) & (c[:-1] < 0.5)).sum()
    out["step_freq_hz"] = float(touchdowns / 2 / (win.sum() * dt))
    swing_h = np.where(c < 0.5, r["foot_h"][win], np.nan)
    out["swing_height"] = float(np.nanmax(swing_h)) if np.isfinite(swing_h).any() else None
    out["base_height"] = float(np.mean(r["z"][win]))
    out["max_tilt_deg"] = float(np.degrees(np.arccos(np.clip(r["up_z"][win].min(), -1, 1))))
    out["peak_torque_ratio"] = float(r["torque_ratio"][win].max())
    out["min_joint_margin"] = float(r["joint_margin"][win].min())
  return out


# ------------------------------------------------------------------ summaries
def _mean(xs):
  xs = [x for x in xs if x is not None and np.isfinite(x)]
  return float(np.mean(xs)) if xs else None


def summarise(metrics: list) -> dict:
  by = lambda g: [m for m in metrics if m["group"] == g]
  out = {}
  grid = by("grid")
  if grid:
    out["grid"] = {
        "survival": _mean([m["survived"] for m in grid]),
        "lin_vel_err": _mean([m.get("lin_vel_err") for m in grid]),
        "yaw_rate_err": _mean([m.get("yaw_rate_err") for m in grid]),
        "cost_of_transport": _mean([m.get("cost_of_transport") for m in grid]),
        "foot_slip": _mean([m.get("foot_slip") for m in grid]),
        "worst": sorted(({"name": m["name"], "err": m.get("lin_vel_err"), "survived": m["survived"]}
                         for m in grid), key=lambda x: (x["survived"], -(x["err"] or 9)))[:5],
    }
  tr = by("training")
  if tr:
    t_alive = sum(m["time_alive_s"] for m in tr)
    falls = sum(not m["survived"] for m in tr)
    out["training"] = {"survival": _mean([m["survived"] for m in tr]),
                       "falls_per_min": falls / (t_alive / 60) if t_alive else None,
                       "lin_vel_err": _mean([m.get("lin_vel_err") for m in tr]),
                       "yaw_rate_err": _mean([m.get("yaw_rate_err") for m in tr])}
  ood = by("ood")
  if ood:
    table = {}
    for m in ood:
      table.setdefault(m["param"], {}).setdefault(m["value"], []).append(m)
    out["ood"] = {p: {str(v): {"survival": _mean([x["survived"] for x in ms]),
                               "lin_vel_err": _mean([x.get("lin_vel_err") for x in ms])}
                      for v, ms in sorted(vals.items())} for p, vals in table.items()}
  slope = by("slope")
  if slope:
    out["slope"] = {}
    for m in slope:
      key = f"{m['deg']} deg {m['dir']}"
      e = out["slope"].setdefault(key, {})
      e[m["cmd"]] = {"survived": m["survived"], "vx": m.get("vx_mean"), "lin_vel_err": m.get("lin_vel_err")}
    ok = [m for m in slope if m["deg"] <= 6]
    out["slope_summary"] = {
        "survival_le6": _mean([m["survived"] for m in ok]),
        "stand_drift_le6": _mean([m.get("lin_vel_err") if m["survived"] else 2.0 for m in ok
                                  if m["cmd"] == "stand"]),
        "walk_err_le6": _mean([m.get("lin_vel_err") if m["survived"] else 2.0 for m in ok
                               if m["cmd"] == "walk"])}
  imu = by("imu")
  if imu:
    out["imu"] = {m["name"]: {"survived": m["survived"], "lin_vel_err": m.get("lin_vel_err")} for m in imu}
    out["imu_summary"] = {"stand_drift": _mean([m.get("lin_vel_err") if m["survived"] else 2.0
                                                for m in imu if m["cmd"] == "stand"])}
  small = by("small")
  if small:
    rows = {}
    for m in small:
      c = np.array([m["vx"], m["vy"], m["wz"]])
      got = np.array([m.get("vx_mean") or 0.0, m.get("vy_mean") or 0.0, m.get("wz_mean") or 0.0])
      k = int(np.argmax(np.abs(c)))
      rows[m["name"]] = {"command": float(c[k]), "achieved": float(got[k]),
                         "fraction": float(got[k] / c[k]) if m["survived"] else 0.0}
    out["small"] = rows
    out["small_summary"] = {"mean_fraction": _mean([r["fraction"] for r in rows.values()]),
                            "min_fraction": min(r["fraction"] for r in rows.values())}
  push = by("push")
  if push:
    res = {}
    for m in push:
      res.setdefault((m["cmd"], m["dir"]), {}).setdefault(m["mag"], []).append(m["survived"])
    out["push"] = {}
    for (cmd, d), mags in res.items():
      # largest magnitude such that it and every smaller one were always survived
      best = 0.0
      for mag, sv in sorted(mags.items()):
        if all(sv):
          best = mag
        else:
          break
      out["push"][f"{cmd}/{d}"] = {"max_recovered_mps": best,
                                    "survival_by_mag": {str(k): _mean(v) for k, v in sorted(mags.items())}}
  return out
