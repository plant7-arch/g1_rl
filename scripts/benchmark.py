"""Phase 0 benchmark: MJX throughput for the G1 on all GPUs + training-time estimate.

Runs the same work an RL rollout does (random policy around the home pose,
5 physics substeps per control step, a small MLP forward pass, auto-reset of
fallen robots) for several batch sizes, then extrapolates wall-clock training
time for typical step budgets.

  python scripts/benchmark.py                      # GPU (all devices), default sweep
  python scripts/benchmark.py --scaling            # also 1-GPU numbers
  python scripts/benchmark.py --platform cpu --envs 8,16 --iters 2   # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402  (must precede jax import)


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  p.add_argument("--envs", default="1024,2048,4096,8192",
                 help="comma-separated TOTAL env counts (split across devices)")
  p.add_argument("--unroll", type=int, default=20, help="control steps per rollout call")
  p.add_argument("--iters", type=int, default=5, help="timed rollout calls per size")
  p.add_argument("--impl", default="jax", choices=["jax", "warp"])
  p.add_argument("--parallel", default="gspmd", choices=["gspmd", "shard_map"])
  p.add_argument("--scaling", action="store_true", help="also benchmark with 1 device")
  p.add_argument("--targets", default="150e6,250e6,400e6",
                 help="env-step budgets to estimate training time for")
  p.add_argument("--overhead", type=float, default=1.3,
                 help="multiplier for obs/reward/DR + PPO update cost over raw sim")
  p.add_argument("--out", default="outputs/benchmark.json")
  return p.parse_args()


def build(rt, args, n_envs):
  import jax
  import jax.numpy as jnp
  import numpy as np

  from g1_rl import parallel, robot, sim
  from g1_rl.config import SimConfig

  cfg = SimConfig(impl=args.impl)
  m = robot.load_mj_model(cfg)
  info = robot.robot_info(m, cfg.home_key)
  mx = robot.make_mjx_model(m, cfg)

  home_q = jnp.asarray(info.home_qpos)
  home_u = jnp.asarray(info.home_ctrl)
  lo, hi = jnp.asarray(info.ctrl_lo), jnp.asarray(info.ctrl_hi)
  qadr = jnp.asarray(info.actuator_qpos_adr)
  n_sub, scale, nu = cfg.n_substeps, cfg.action_scale, info.nu

  # Stand-in policy with the planned network size (obs ~ 100 -> 512-256-128 -> 29).
  obs_dim = 3 + 3 + nu + nu + nu + 3 + 2
  rng = np.random.default_rng(0)
  sizes = [obs_dim, 512, 256, 128, nu]
  W = [jnp.asarray(rng.normal(0, 1 / np.sqrt(a), (a, b)), jnp.float32)
       for a, b in zip(sizes[:-1], sizes[1:])]

  def policy(obs):
    h = obs
    for w in W[:-1]:
      h = jax.nn.elu(h @ w)
    return jnp.tanh(h @ W[-1])

  packer = sim.DataPacker(mx)

  def init_one(key):
    noise = jax.random.uniform(key, (nu,), minval=-0.05, maxval=0.05)
    q = home_q.at[qadr].add(noise)
    d = sim.init_data(mx, q, home_u)
    return packer.pack(d), jnp.zeros((nu,))

  def env_step(carry, _, d_home):
    d, last_a, key = carry
    key, k = jax.random.split(key)
    obs = jnp.concatenate([
        d.qvel[3:6], d.qpos[3:6], d.qpos[qadr] - home_u, d.qvel[6:6 + nu] * 0.05,
        last_a, jnp.zeros(3), jnp.zeros(2)])
    a = policy(obs) + 0.3 * jax.random.normal(k, (nu,))
    ctrl = jnp.clip(home_u + scale * a, lo, hi)
    d = sim.step_ctrl(mx, d, ctrl, n_sub)
    done = sim.is_unhealthy(d)
    d = sim.tree_where(done, d_home, d)
    return (d, a, key), done

  def rollout_one(packed, last_a, key):
    d = packer.unpack(packed)
    d_home = sim.init_data(mx, home_q, home_u)  # reset target (constant-folded by XLA)
    (d, last_a, key), done = jax.lax.scan(
        lambda c, x: env_step(c, x, d_home), (d, last_a, key), None, length=args.unroll)
    return packer.pack(d), last_a, key, done.sum()

  if args.parallel == "shard_map":
    B = parallel.BATCH
    init_fn = jax.jit(parallel.shard_map(jax.vmap(init_one), rt.mesh, in_specs=B, out_specs=(B, B)))
    roll_fn = jax.jit(
        parallel.shard_map(jax.vmap(rollout_one), rt.mesh, in_specs=(B, B, B),
                           out_specs=(B, B, B, B)),
        donate_argnums=(0, 1, 2))
  else:  # gspmd (default)
    init_fn = parallel.jit_batched(jax.vmap(init_one), rt.mesh)
    roll_fn = parallel.jit_batched(jax.vmap(rollout_one), rt.mesh, donate_argnums=(0, 1, 2))
  return m, info, init_fn, roll_fn


def bench_size(rt, args, n_envs):
  import jax

  from g1_rl import parallel
  from g1_rl.config import SimConfig

  parallel.check_divisible(n_envs, rt.mesh)
  m, info, init_fn, roll_fn = build(rt, args, n_envs)
  keys = parallel.put_batched(jax.random.split(jax.random.PRNGKey(0), n_envs), rt.mesh)
  d, last_a = init_fn(keys)

  t0 = time.perf_counter()
  compiled = roll_fn.lower(d, last_a, keys).compile()
  compile_s = time.perf_counter() - t0
  collectives = parallel.count_collectives_text(compiled.as_text()) if rt.n_devices > 1 else {}
  roll_fn = compiled
  d, last_a, keys, falls = roll_fn(d, last_a, keys)  # warm-up
  jax.block_until_ready(falls)

  total_falls = 0
  t0 = time.perf_counter()
  for _ in range(args.iters):
    d, last_a, keys, falls = roll_fn(d, last_a, keys)
  total_falls = int(jax.device_get(falls).sum())  # one host sync at the end
  jax.block_until_ready(d)
  dt = time.perf_counter() - t0

  jnp = jax.numpy
  finite = all(bool(jnp.isfinite(l).all()) for l in d if jnp.issubdtype(l.dtype, jnp.floating))
  env_steps = n_envs * args.unroll * args.iters
  mem = None
  try:
    stats = [dv.memory_stats() or {} for dv in rt.devices]
    mem = max(s.get("peak_bytes_in_use", 0) for s in stats) / 2**30
  except Exception:
    pass
  return dict(n_envs=n_envs, n_devices=rt.n_devices, compile_s=round(compile_s, 1),
              env_steps_per_s=env_steps / dt,
              physics_steps_per_s=env_steps * SimConfig().n_substeps / dt,
              falls_last_call=total_falls, finite=finite, collectives=collectives,
              peak_mem_gib_per_device=None if mem is None else round(mem, 2))


def run_sweep(rt, args, env_list):
  rows = []
  for n in env_list:
    if n % rt.n_devices:
      continue
    try:
      r = bench_size(rt, args, n)
    except Exception as e:  # typically RESOURCE_EXHAUSTED (OOM) at large batch
      msg = str(e).splitlines()[0][:160]
      print(f"  n_envs={n:6d}  FAILED: {msg}")
      if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
        break
      raise
    rows.append(r)
    mem = f"{r['peak_mem_gib_per_device']:.2f} GiB/dev" if r["peak_mem_gib_per_device"] else "n/a"
    print(f"  n_envs={n:6d}  {r['env_steps_per_s']:>10,.0f} env-steps/s  "
          f"({r['physics_steps_per_s']:>11,.0f} phys/s)  compile {r['compile_s']:5.1f}s  "
          f"mem {mem}  finite={r['finite']}"
          + (f"  collectives={r['collectives']}" if r["collectives"] else ""))
  return rows


def estimate(best, targets, overhead):
  sps = best["env_steps_per_s"] / overhead
  out = []
  for t in targets:
    hours = t / sps / 3600
    out.append(dict(env_steps=t, hours=round(hours, 2),
                    kaggle_sessions_12h=int(-(-hours // 11.5))))
  return out


def main():
  args = parse_args()
  rt = device.setup(args.platform)
  import jax
  import mujoco

  print(f"jax {jax.__version__} | mujoco {mujoco.__version__} | {rt.describe()} | impl={args.impl} | parallel={args.parallel}")
  env_list = [int(x) for x in args.envs.split(",")]
  targets = [float(x) for x in args.targets.split(",")]

  results = {"platform": rt.platform, "devices": rt.describe(), "jax": jax.__version__,
             "mujoco": mujoco.__version__, "unroll": args.unroll, "sweeps": {}}

  configs = [rt]
  if args.scaling and rt.n_devices > 1:
    configs.insert(0, device.setup(args.platform, max_devices=1))
  for r in configs:
    print(f"\n== {r.n_devices} device(s) ==")
    results["sweeps"][str(r.n_devices)] = run_sweep(r, args, env_list)

  rows = results["sweeps"][str(rt.n_devices)]
  if not rows:
    print("No successful runs.")
    return
  best = max(rows, key=lambda r: r["env_steps_per_s"])
  results["best"] = best
  results["estimate"] = estimate(best, targets, args.overhead)
  print(f"\nBest: {best['n_envs']} envs -> {best['env_steps_per_s']:,.0f} env-steps/s "
        f"on {rt.n_devices} device(s)")
  print(f"Estimated training wall-clock (overhead x{args.overhead} for env logic + PPO update):")
  for e in results["estimate"]:
    print(f"  {e['env_steps'] / 1e6:5.0f}M env steps -> {e['hours']:6.2f} h "
          f"(~{e['kaggle_sessions_12h']} Kaggle session(s))")

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nSaved {args.out}")


if __name__ == "__main__":
  main()
