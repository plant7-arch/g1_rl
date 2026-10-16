"""Evaluate a trained policy: MJX suite on GPU (CPU fallback) + sim-to-sim in CPU MuJoCo.

  python scripts/evaluate.py --run outputs/runs/g1_walk_v1          # ~4-5 min on Kaggle 2xT4
  python scripts/evaluate.py --policy path/to/policy_final.npz --quick
  python scripts/evaluate.py --run ... --platform cpu --quick         # no GPU

Writes <out>/report.md (read this), report.json, eval_summary.png and videos:
  mjx_forward_0p5.mp4            nominal robot walking forward (training simulator)
  full_model_forward_0p5.mp4     the same in the full-mesh MuJoCo model
  full_model_walk_push_left_0p5.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402


def find_policy(args):
  if args.policy:
    return args.policy
  for name in ("policy_final.npz", "policy_latest.npz"):
    p = os.path.join(args.run, name)
    if os.path.exists(p):
      return p
  raise SystemExit(f"no policy_final.npz / policy_latest.npz in {args.run}")


def main():
  p = argparse.ArgumentParser()
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--run", help="training run dir (uses policy_final.npz, else policy_latest.npz)")
  src.add_argument("--policy", help="exported policy .npz")
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  p.add_argument("--quick", action="store_true", help="smaller suite, 10 s scenarios")
  p.add_argument("--seconds", type=float, default=None, help="MJX scenario length (default 20, quick 10)")
  p.add_argument("--no-obs-noise", action="store_true", help="evaluate with perfect sensors")
  p.add_argument("--no-mjx", action="store_true")
  p.add_argument("--no-sim2sim", action="store_true")
  p.add_argument("--no-video", action="store_true")
  p.add_argument("--out", default=None)
  args = p.parse_args()

  rt = device.setup(args.platform)
  from g1_rl import evaluate, render, report, sim2sim
  from g1_rl.config import SimConfig
  from g1_rl.policy import Policy

  path = find_policy(args)
  pi = Policy.load(path, device=rt.devices[0])
  run_name = os.path.basename(os.path.normpath(args.run)) if args.run else os.path.splitext(os.path.basename(path))[0]
  out = args.out or os.path.join("outputs", "eval", run_name)
  os.makedirs(out, exist_ok=True)
  seconds = args.seconds or (10.0 if args.quick else 20.0)
  meta = {"policy": path, "iteration": pi.meta.get("iteration"), "env_steps": pi.meta.get("env_steps", 0),
          "runtime": rt.describe(), "obs_noise": not args.no_obs_noise, "seconds": seconds,
          "quick": args.quick}
  print(f"policy {path} (iteration {meta['iteration']}) | {rt.describe()} | out {out}")

  summary = metrics = s2s = None
  if not args.no_mjx:
    t0 = time.time()
    scenarios = evaluate.build_suite(args.quick)
    fwd = next(i for i, s in enumerate(scenarios) if s.group == "grid" and s.command == (0.5, 0.0, 0.0))
    print(f"MJX suite: {len(scenarios)} scenarios x {seconds:.0f} s in one batch (compiling first)...", flush=True)
    metrics, extra = evaluate.run_mjx_suite(pi, rt, scenarios, seconds=seconds,
                                            obs_noise=not args.no_obs_noise, record_qpos_for=(fwd,))
    summary = evaluate.summarise(metrics)
    print(f"  done in {time.time() - t0:.0f}s")
    if not args.no_video:
      # stop 0.5 s after a fall (the env auto-resets the robot afterwards)
      alive_steps = int(metrics[fwd]["time_alive_s"] / 0.02)
      q = extra["qpos"][fwd][: min(int(10 / 0.02), alive_steps + 25)]
      traj = render.save_trajectory(os.path.join(out, "mjx_forward_0p5.npz"), q, fps=50,
                                    xml_path=SimConfig().xml_path)
      render.try_render(traj, os.path.join(out, "mjx_forward_0p5.mp4"))

  if not args.no_sim2sim:
    t0 = time.time()
    print("Sim-to-sim in CPU MuJoCo (training model, full mesh model)...", flush=True)
    s2s = sim2sim.run_sim2sim(
        pi, {"training_model_cpu": SimConfig().xml_path, "full_model": SimConfig().eval_xml_path},
        video_dir=None if args.no_video else out,
        video_scenarios=("forward 0.5", "walk, push left 0.5"), video_models=("full_model",))
    print(f"  done in {time.time() - t0:.0f}s")

  rep = report.write(out, meta, summary, metrics, s2s)
  print()
  print(open(os.path.join(out, "report.md")).read())
  print(f"files in {out}: {sorted(os.listdir(out))}")
  res = [c["pass"] for c in rep["checks"]]
  print(f"readiness checks: {res.count(True)} passed, {res.count(False)} failed, "
        f"{res.count(None)} not measured (of {len(res)})")


if __name__ == "__main__":
  main()
