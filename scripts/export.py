"""Phase 6: export a trained policy as a deployment bundle (ONNX + npz + standalone runner).

  python scripts/export.py --run outputs/runs/g1_walk_v1                # -> outputs/export/g1_walk_v1/
  python scripts/export.py --policy path/to/policy_final.npz --out my_bundle --include-model

Checks, in order (all on CPU, < 1 min):
  1. ONNX graph == NumPy reference == JAX policy on 4096 random observations
  2. closed loop: the ONNX policy walks 10 s at 0.5 m/s in the full mesh MuJoCo model
     through the g1_rl sim-to-sim runner, and the standalone deploy/run_mujoco.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
  p = argparse.ArgumentParser()
  src = p.add_mutually_exclusive_group(required=True)
  src.add_argument("--run", help="training run dir (uses policy_final.npz, else policy_latest.npz)")
  src.add_argument("--policy", help="exported policy .npz")
  p.add_argument("--out", default=None)
  p.add_argument("--include-model", action="store_true",
                 help="copy the MuJoCo Menagerie unitree_g1 model (38 MB) into the bundle")
  p.add_argument("--tol", type=float, default=1e-4, help="max allowed |ONNX - reference| action difference")
  args = p.parse_args()

  if args.run:
    cands = [os.path.join(args.run, f) for f in ("policy_final.npz", "policy_latest.npz")]
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
      raise SystemExit(f"no policy_final.npz / policy_latest.npz in {args.run}")
    name = os.path.basename(os.path.normpath(args.run))
  else:
    path, name = args.policy, os.path.splitext(os.path.basename(args.policy))[0]
  out = args.out or os.path.join("outputs", "export", name)

  from g1_rl import device
  rt = device.setup("cpu")  # tiny network: CPU is the reference platform for the check
  from g1_rl import export
  from g1_rl.config import SimConfig
  from g1_rl.policy import Policy
  from g1_rl.sim2sim import MujocoRunner

  assets = os.path.dirname(SimConfig().eval_xml_path)
  info = export.export_bundle(path, out, include_model=assets if args.include_model else None,
                              jax_policy=Policy.load(path, device=rt.devices[0]))
  c = info["check"]
  print(f"exported {path}\n  -> {out}: {sorted(os.listdir(out))}")
  print(f"1. max |action difference| on {c['samples']} observations: ONNX vs NumPy {c['onnx_vs_numpy_action']:.1e}, "
        f"JAX vs NumPy {c['jax_vs_numpy_action']:.1e}, targets {c['onnx_vs_numpy_targets']:.1e} rad")
  ok = max(c["onnx_vs_numpy_action"], c["jax_vs_numpy_action"]) <= args.tol

  pi = export.OnnxPolicy(os.path.join(out, "policy.onnx"), providers=["CPUExecutionProvider"])
  r = MujocoRunner(SimConfig().eval_xml_path, pi).run((0.5, 0.0, 0.0), seconds=10.0)
  num = lambda v: "n/a" if v is None else f"{v:.2f}"  # None when the robot fell (v1.0 fix)
  print(f"2. ONNX policy in the full mesh model, 10 s at 0.5 m/s: survived {r['survived']}, "
        f"mean vx {num(r['vx_mean'])} m/s, velocity error {num(r['lin_vel_err'])} m/s")
  ok &= bool(r["survived"])
  res = subprocess.run([sys.executable, os.path.join(out, "run_mujoco.py"), "--xml", SimConfig().eval_xml_path,
                        "--cmd", "0.5", "0", "0", "--seconds", "10"], capture_output=True, text=True)
  print("   standalone run_mujoco.py: " + (res.stdout.strip().splitlines() or ["(no output)"])[-1])
  ok &= res.returncode == 0
  info["closed_loop"] = {"sim2sim_full_model": {k: v for k, v in r.items() if k != "qpos"},
                         "standalone_runner_ok": res.returncode == 0}
  with open(os.path.join(out, "export_info.json"), "w") as f:
    json.dump(info, f, indent=2)
  print("export OK" if ok else "EXPORT CHECK FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
