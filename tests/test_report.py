"""Phase 5: suite summaries and readiness checks on hand-made results (no simulation)."""

import os

from g1_rl import evaluate, report


def m(group, survived=True, **kw):
  base = {"group": group, "name": "x", "survived": survived, "time_alive_s": 20.0 if survived else 3.0,
          "lin_vel_err": 0.1, "yaw_rate_err": 0.1, "vx_mean": 0.5, "vy_mean": 0.0, "wz_mean": 0.0,
          "cost_of_transport": 0.5, "foot_slip": 0.02}
  base.update(kw)
  return base


def test_push_recovery_is_largest_magnitude_with_no_failure_below_it():
  ms = []
  for mag, ok in [(0.25, True), (0.5, True), (0.75, False), (1.0, True)]:  # 1.0 survived by luck
    ms.append(m("push", ok, cmd="walk", dir="left", mag=mag))
  s = evaluate.summarise(ms)
  assert s["push"]["walk/left"]["max_recovered_mps"] == 0.5


def test_checks_pass_fail_and_unmeasured(tmp_path):
  grid = [m("grid", vx=v, vy=0.0, wz=0.0) for v in (0.0, 0.5)]
  push = [m("push", True, cmd=c, dir=d, mag=0.5) for c in ("walk", "stand")
          for d in ("front", "back", "left", "right")]
  train = [m("training", True), m("training", False)]  # 50 % survival -> fails
  ms = grid + push + train
  s = evaluate.summarise(ms)
  rep = report.write(str(tmp_path), {"policy": "p", "env_steps": 0}, s, ms, None)
  by = {c["check"]: c["pass"] for c in rep["checks"]}
  assert by["Nominal: survives every grid command"] is True
  assert by["Nominal: linear velocity error (m/s)"] is True
  assert by["Training conditions: survival over the whole scenario"] is False
  assert by["Walking: recovers from 0.5 m/s pushes in all directions"] is True
  assert by["Sim-to-sim (full mesh model): all scenarios survived"] is None  # not run
  assert s["training"]["falls_per_min"] == 1 / ((20 + 3) / 60)
  for f in ("report.json", "report.md", "eval_summary.png"):
    assert os.path.exists(tmp_path / f)
  assert "❌" in open(tmp_path / "report.md").read()
