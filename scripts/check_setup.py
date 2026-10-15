"""Phase 0 sanity check: versions, devices, model, and MJX-vs-MuJoCo agreement.

  python scripts/check_setup.py             # GPU by default, CPU fallback
  python scripts/check_setup.py --platform cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  p.add_argument("--steps", type=int, default=100, help="physics steps for the agreement test")
  args = p.parse_args()

  rt = device.setup(args.platform)
  import jax
  import jax.numpy as jnp
  import mujoco
  import numpy as np
  from mujoco import mjx

  from g1_rl import robot, sim
  from g1_rl.config import SimConfig

  print(f"python {sys.version.split()[0]} | jax {jax.__version__} | mujoco {mujoco.__version__}")
  print(f"runtime: {rt.describe()}")
  for d in rt.devices:
    try:
      s = d.memory_stats() or {}
      print(f"  {d}: {s.get('bytes_limit', 0) / 2**30:.1f} GiB usable")
    except Exception:
      pass

  cfg = SimConfig()
  m = robot.load_mj_model(cfg)
  info = robot.robot_info(m, cfg.home_key)
  print(robot.summary(m, info))
  print(f"  actuators: {info.nu} (legs {len(info.leg_idx)}, waist {len(info.waist_idx)}, "
        f"arms {len(info.arm_idx)}), control {1 / cfg.ctrl_dt:.0f} Hz, {cfg.n_substeps} substeps")

  # 1) Classic MuJoCo (CPU, float64) reference: hold the home pose for N steps.
  d_ref = mujoco.MjData(m)
  d_ref.qpos[:] = info.home_qpos
  d_ref.ctrl[:] = info.home_ctrl
  mujoco.mj_forward(m, d_ref)
  for _ in range(args.steps):
    mujoco.mj_step(m, d_ref)

  # 2) MJX on each device (float32).
  for dev in rt.devices:
    with jax.default_device(dev):
      mx = mjx.put_model(m, impl=cfg.impl)
      init = jax.jit(lambda q, u: sim.init_data(mx, q, u))
      d = init(jnp.asarray(info.home_qpos), jnp.asarray(info.home_ctrl))
      step = jax.jit(lambda d: sim.step_ctrl(mx, d, d.ctrl, cfg.n_substeps))
      t0 = time.perf_counter()
      d = step(d)
      jax.block_until_ready(d.qpos)
      c = time.perf_counter() - t0
      for _ in range(args.steps // cfg.n_substeps - 1):
        d = step(d)
      q = np.asarray(d.qpos)
    err = np.abs(q - d_ref.qpos)
    ok = np.isfinite(q).all() and err[:3].max() < 0.02 and err[7:].max() < 0.05
    print(f"  {dev}: compile {c:.1f}s | pelvis z mjx={q[2]:.3f} ref={d_ref.qpos[2]:.3f} | "
          f"max|dq| joints={err[7:].max():.4f} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
      sys.exit(1)
  print("Setup check passed.")


if __name__ == "__main__":
  main()
