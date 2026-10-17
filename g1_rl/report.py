"""Evaluation report: report.json (everything), report.md (readable), eval_summary.png."""

from __future__ import annotations

import json
import os

import numpy as np

# Guidance thresholds for "ready for the next step" (sim-to-real needs all of them
# and more; they are indicators, not guarantees).
CHECKS = [
    ("Nominal: survives every grid command", lambda s, x: _get(s, "grid", "survival"), ">=", 0.98),
    ("Nominal: linear velocity error (m/s)", lambda s, x: _get(s, "grid", "lin_vel_err"), "<=", 0.20),
    ("Nominal: yaw-rate error (rad/s)", lambda s, x: _get(s, "grid", "yaw_rate_err"), "<=", 0.25),
    ("Training conditions: survival over the whole scenario", lambda s, x: _get(s, "training", "survival"),
     ">=", 0.90),
    ("Walking: recovers from 0.5 m/s pushes in all directions",
     lambda s, x: _min_push(s, "walk"), ">=", 0.5),
    ("Standing: recovers from 0.5 m/s pushes in all directions",
     lambda s, x: _min_push(s, "stand"), ">=", 0.5),
    ("Sim-to-sim (CPU MuJoCo, training model): all scenarios survived",
     lambda s, x: _s2s_survival(x, "training_model_cpu"), ">=", 1.0),
    ("Sim-to-sim (full mesh model): all scenarios survived",
     lambda s, x: _s2s_survival(x, "full_model"), ">=", 1.0),
    # v0.9: the weak spots found in the v3 test campaign
    ("Slopes up to 6 deg (up, down, across): survival", lambda s, x: _get(s, "slope_summary", "survival_le6"),
     ">=", 1.0),
    ("Standing on slopes up to 6 deg: drift speed (m/s)", lambda s, x: _get(s, "slope_summary", "stand_drift_le6"),
     "<=", 0.15),
    ("IMU mounted 3 deg off: standing drift speed (m/s)", lambda s, x: _get(s, "imu_summary", "stand_drift"),
     "<=", 0.15),
    ("Small commands (0.15-0.2 m/s, 0.2-0.3 rad/s): mean fraction achieved",
     lambda s, x: _get(s, "small_summary", "mean_fraction"), ">=", 0.6),
]


def _get(s, *keys):
  for k in keys:
    if not isinstance(s, dict) or k not in s:
      return None
    s = s[k]
  return s


def _min_push(s, cmd):
  p = _get(s, "push")
  if not p:
    return None
  vals = [v["max_recovered_mps"] for k, v in p.items() if k.startswith(cmd + "/")]
  return min(vals) if vals else None


def _s2s_survival(x, label):
  r = (x or {}).get(label)
  if not r:
    return None
  runs = [v for k, v in r.items() if not k.startswith("_")]
  return float(np.mean([v["survived"] for v in runs])) if runs else None


def _fmt(v, nd=2):
  if v is None:
    return "–"
  if isinstance(v, bool):
    return "yes" if v else "NO"
  if isinstance(v, float):
    return f"{v:.{nd}f}"
  return str(v)


def checks(summary, sim2sim):
  rows = []
  for name, fn, op, thr in CHECKS:
    v = fn(summary, sim2sim)
    ok = None if v is None else (v >= thr if op == ">=" else v <= thr)
    rows.append({"check": name, "value": v, "threshold": f"{op} {thr}", "pass": ok})
  return rows


def write(out_dir: str, meta: dict, summary: dict | None, metrics: list | None,
          sim2sim: dict | None) -> dict:
  os.makedirs(out_dir, exist_ok=True)
  chk = checks(summary or {}, sim2sim or {})
  report = {"meta": meta, "checks": chk, "mjx_summary": summary, "sim2sim": sim2sim,
            "mjx_scenarios": metrics}
  with open(os.path.join(out_dir, "report.json"), "w") as f:
    json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
  md = _markdown(meta, chk, summary, sim2sim)
  with open(os.path.join(out_dir, "report.md"), "w") as f:
    f.write(md)
  if summary and metrics:
    try:
      _plot(os.path.join(out_dir, "eval_summary.png"), summary, metrics)
    except Exception as e:  # plotting must never lose the numbers
      print(f"[report] plot skipped: {e}")
  return report


def _markdown(meta, chk, s, x) -> str:
  L = [f"# Evaluation: {meta.get('policy')}", "",
       f"policy iteration {meta.get('iteration')}, {meta.get('env_steps', 0) / 1e6:.0f}M training steps · "
       f"{meta.get('runtime')} · obs noise {'on' if meta.get('obs_noise') else 'off'} · "
       f"{meta.get('seconds')} s per MJX scenario", "",
       "## Readiness checks", "", "| | check | value | target |", "|---|---|---|---|"]
  for c in chk:
    mark = "–" if c["pass"] is None else ("✅" if c["pass"] else "❌")
    L.append(f"| {mark} | {c['check']} | {_fmt(c['value'])} | {c['threshold']} |")
  if s and "grid" in s:
    g = s["grid"]
    L += ["", "## Nominal robot, fixed commands (MJX)", "",
          f"survival {_fmt(g['survival'])} · linear velocity error {_fmt(g['lin_vel_err'])} m/s · "
          f"yaw-rate error {_fmt(g['yaw_rate_err'])} rad/s · cost of transport {_fmt(g['cost_of_transport'])} · "
          f"stance-foot slip {_fmt(g['foot_slip'])} m/s", "", "Worst commands: " +
          ", ".join(f"{w['name']} ({'fell' if not w['survived'] else _fmt(w['err']) + ' m/s'})"
                    for w in g["worst"])]
  if s and "training" in s:
    t = s["training"]
    L += ["", "## Training conditions: full randomisation, random commands and pushes (MJX)", "",
          f"survival {_fmt(t['survival'])} · falls per minute {_fmt(t['falls_per_min'])} · "
          f"linear velocity error {_fmt(t['lin_vel_err'])} m/s · yaw-rate error {_fmt(t['yaw_rate_err'])} rad/s"]
  if s and "push" in s:
    L += ["", "## Push recovery: largest push always survived (m/s, single kick)", "",
          "| | front | back | left | right |", "|---|---|---|---|---|"]
    for cmd in ("stand", "walk"):
      L.append(f"| {cmd} | " + " | ".join(_fmt(s["push"].get(f"{cmd}/{d}", {}).get("max_recovered_mps"))
                                          for d in ("front", "back", "left", "right")) + " |")
  if s and "slope" in s:
    L += ["", "## Slopes (gravity tilted; robot faces +x): survived / speed error (m/s)", "",
          "| slope | standing | walking 0.5 m/s |", "|---|---|---|"]
    for k, v in s["slope"].items():
      cell = lambda c: (f"{_fmt(v[c]['survived'])} / {_fmt(v[c]['lin_vel_err'])}" if c in v else "–")
      L.append(f"| {k} | {cell('stand')} | {cell('walk')} |")
  if s and "imu" in s:
    L += ["", "## IMU mounted 3 deg off: survived / speed error (m/s)", ""]
    L += [f"- {k}: {_fmt(v['survived'])} / {_fmt(v['lin_vel_err'])}" for k, v in s["imu"].items()]
  if s and "small" in s:
    L += ["", "## Small commands: commanded → achieved (fraction)", ""]
    L += [f"- {k}: {_fmt(v['command'])} → {_fmt(v['achieved'])} ({_fmt(v['fraction'])})"
          for k, v in s["small"].items()]
  if s and "ood" in s:
    L += ["", "## Beyond the training ranges, walking 0.5 m/s: survival (tracking error m/s)", ""]
    train_rng = {"friction": "0.3–1.2", "payload": "−1–8 kg", "kp_scale": "0.75–1.25", "kd_scale": "0.7–1.3",
                 "com_offset_x": "±0.03 m", "frictionloss_scale": "0.5–2", "latency": "0–2 steps"}
    for p, vals in s["ood"].items():
      cells = ", ".join(f"{v}: {_fmt(r['survival'])} ({_fmt(r['lin_vel_err'])})" for v, r in vals.items())
      L.append(f"- **{p}** (trained {train_rng.get(p, '?')}): {cells}")
  if x:
    labels = list(x)
    L += ["", "## Sim-to-sim in CPU MuJoCo: survived / linear velocity error (m/s)", "",
          "| scenario | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    names = [k for k in x[labels[0]] if not k.startswith("_")]
    for n in names:
      row = []
      for lab in labels:
        r = x[lab].get(n, {})
        row.append(f"{_fmt(r.get('survived'))} / {_fmt(r.get('lin_vel_err'))}")
      L.append(f"| {n} | " + " | ".join(row) + " |")
    L += [""] + [f"- **{lab}**: " + ", ".join(f"{k} {v}" for k, v in x[lab]["_model"].items())
                 for lab in labels]
  L.append("")
  return "\n".join(L)


def _plot(path, s, metrics):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(2, 2, figsize=(13, 9))
  grid = [m for m in metrics if m["group"] == "grid" and "vx_mean" in m]
  a = ax[0, 0]
  for wz, mk in ((-0.8, "v"), (0.0, "o"), (0.8, "^")):
    pts = [m for m in grid if m["wz"] == wz and m["vy"] == 0.0]
    a.scatter([m["vx"] for m in pts], [m["vx_mean"] for m in pts], marker=mk,
              c=["tab:blue" if m["survived"] else "tab:red" for m in pts], label=f"wz cmd {wz:+.1f}")
  lim = [-0.7, 1.2]
  a.plot(lim, lim, "k--", lw=0.8)
  a.set(xlabel="commanded vx (m/s)", ylabel="achieved vx (m/s)", title="Forward speed tracking (red = fell)")
  a.legend(fontsize=8)
  a.grid(alpha=0.3)
  a = ax[0, 1]
  a.scatter([m["wz"] for m in grid], [m["wz_mean"] for m in grid], c=[m["vx"] for m in grid], cmap="viridis")
  a.plot([-1, 1], [-1, 1], "k--", lw=0.8)
  a.set(xlabel="commanded yaw rate (rad/s)", ylabel="achieved (rad/s)", title="Turning (colour = vx command)")
  a.grid(alpha=0.3)
  a = ax[1, 0]
  ood = s.get("ood", {})
  for p, vals in ood.items():
    a.plot(range(len(vals)), [v["survival"] for v in vals.values()], marker="o", label=p)
  a.set(xlabel="value index (low → high)", ylabel="survival", ylim=(-0.05, 1.05),
        title="Beyond training ranges (see report.md for values)")
  if ood:
    a.legend(fontsize=7)
  else:
    a.text(0.5, 0.5, "not run", ha="center", va="center", transform=a.transAxes, color="0.5")
  a.grid(alpha=0.3)
  a = ax[1, 1]
  dirs = ("front", "back", "left", "right")
  w = 0.38
  for j, cmd in enumerate(("stand", "walk")):
    vals = [s.get("push", {}).get(f"{cmd}/{d}", {}).get("max_recovered_mps", 0) for d in dirs]
    a.bar(np.arange(4) + (j - 0.5) * w, vals, w, label=cmd)
  a.set_xticks(range(4), dirs)
  a.set(ylabel="largest push survived (m/s)", title="Push recovery")
  a.legend()
  a.grid(alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(path, dpi=110)
