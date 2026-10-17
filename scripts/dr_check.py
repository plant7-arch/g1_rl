"""Phase 3 check: domain randomisation inside the simulator (~3-4 min on Kaggle 2xT4).

One compiled rollout is run twice: with the nominal robot (level 0) and with full
randomisation (level 1). The level lives in the env state, so both runs share
the compilation. Pushes are made frequent here (every 0.5-1 s) so their effect
is measurable within a few seconds of simulated time.

Reports the sampled parameter spread, the base-velocity jump caused by pushes,
fall rates, and the throughput cost of per-robot physics.

  python scripts/dr_check.py
  python scripts/dr_check.py --platform cpu --envs 8 --calls 2 --length 50
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  p.add_argument("--envs", type=int, default=1024)
  p.add_argument("--length", type=int, default=100)
  p.add_argument("--calls", type=int, default=3)
  p.add_argument("--action-std", type=float, default=0.3)
  p.add_argument("--out", default="outputs/phase3_dr.json")
  args = p.parse_args()

  rt = device.setup(args.platform)
  import jax
  import jax.numpy as jnp
  import numpy as np

  from g1_rl.config import DomainRandConfig, EnvConfig
  from g1_rl.env import G1Env
  from g1_rl.vec_env import VecEnv

  cfg = EnvConfig(dr=DomainRandConfig(push_interval=(0.5, 1.0)))
  env = G1Env(cfg)
  venv = VecEnv(env, rt.mesh, args.envs)
  print(f"runtime: {rt.describe()} | envs={args.envs} | pushes every 0.5-1 s for this check")

  def policy(obs, key):
    return args.action_std * jax.random.normal(key, (env.action_size,))

  def record(s):
    return dict(v_xy=env.packer.unpack(s.data).qvel[0:2], push=s.metrics["debug/pushes"],
                step=s.info["step"])

  rollout = venv.make_rollout(policy, args.length, record)

  def run(level, seed):
    state = venv.reset(seed)
    state = env.set_dr_level(state, level)
    # The first episode was sampled at the default level; restart every env at `level`
    # by re-sampling its parameters directly.
    info = dict(state.info)
    keys = jax.random.split(jax.random.PRNGKey(seed + 100), args.envs)
    info["dr"] = jax.vmap(lambda k: env.dr.sample(k, level))(keys)
    state = state._replace(info=info)
    from g1_rl import parallel
    state = parallel.put_batched(state, rt.mesh)
    dr0 = jax.device_get(state.info["dr"])
    keys = venv.keys(seed + 1)
    recs, sums, t_run = [], None, 0.0
    for i in range(args.calls):
      t0 = time.perf_counter()
      state, keys, rec, met = rollout(state, keys)
      jax.block_until_ready(rec)
      if i > 0:
        t_run += time.perf_counter() - t0
      recs.append(jax.device_get(rec))
      met = {k: float(np.asarray(v).sum()) for k, v in jax.device_get(met).items()}
      sums = met if sums is None else {k: sums[k] + met[k] for k in sums}
    cat = lambda k: np.concatenate([r[k] for r in recs], axis=1)
    v, push, step = cat("v_xy"), cat("push"), cat("step")
    dv = np.linalg.norm(v[:, 1:] - v[:, :-1], axis=-1)
    same_ep = step[:, 1:] == step[:, :-1] + 1
    pushed = (push[:, 1:] > 0) & same_ep
    calm = (push[:, 1:] == 0) & same_ep
    timed = args.envs * args.length * max(args.calls - 1, 1)
    return dict(
        dr=dr0, pushes=int(push.sum()),
        dv_push=float(dv[pushed].mean()) if pushed.any() else float("nan"),
        dv_calm=float(dv[calm].mean()),
        falls=sums["episode/fell"], episodes=sums["episode/count"],
        mean_len=sums["episode/length"] / max(sums["episode/count"], 1),
        sps=timed / t_run if args.calls > 1 else None)

  t0 = time.perf_counter()
  nominal = run(0.0, 0)
  t_compile = time.perf_counter() - t0
  full = run(1.0, 10)
  d = full["dr"]

  def spread(x):
    x = np.asarray(x, np.float64).ravel()
    return f"{x.min():6.2f} .. {x.max():6.2f} (mean {x.mean():5.2f})"

  print(f"\nSampled robots at level 1 ({args.envs} envs):")
  print(f"  floor friction      {spread(d['friction'])}")
  print(f"  torso payload (kg)  {spread(d['payload'])}")
  print(f"  link mass scale     {spread(d['mass_scale'][:, 1:])}")
  print(f"  kp scale            {spread(d['kp_scale'])}")
  print(f"  kd scale            {spread(d['kd_scale'])}")
  print(f"  torso CoM offset (m){spread(d['com_offset'])}")
  print(f"  action latency      1 step: {np.mean(d['latency'] == 1):.0%}, 2 steps: {np.mean(d['latency'] == 2):.0%}")
  tilt = np.degrees(np.arccos(-np.asarray(d["gravity_dir"])[:, 2]))
  print(f"  slope (deg)         {spread(tilt)}, on a slope: {np.mean(tilt > 1e-3):.0%}")
  print(f"  IMU tilt (deg)      {spread(np.degrees(d['imu_tilt']))}")
  print(f"  gyro bias (rad/s)   {spread(d['gyro_bias'])}")
  print(f"\n{'':22}{'nominal (0)':>14}{'randomised (1)':>16}")
  for k, label, fmt in [("pushes", "push events", "{:.0f}"),
                        ("dv_push", "|dv| on push step", "{:.3f}"), ("dv_calm", "|dv| other steps", "{:.3f}"),
                        ("falls", "falls", "{:.0f}"), ("mean_len", "mean episode length", "{:.0f}"),
                        ("sps", "env-steps/s", "{:.0f}")]:
    a, b = nominal[k], full[k]
    fa = fmt.format(a) if a is not None else "n/a"
    fb = fmt.format(b) if b is not None else "n/a"
    print(f"  {label:<20}{fa:>14}{fb:>16}")
  print(f"(first run incl. compile: {t_compile:.0f}s)")

  lat = float(np.mean(d["latency"] > 0))
  checks = {
      "level 0 is the nominal robot": bool(np.allclose(nominal["dr"]["friction"], 1.0)
                                          and np.allclose(nominal["dr"]["kp_scale"], 1.0)
                                          and not np.any(nominal["dr"]["latency"])),
      "level 1 parameters vary across robots": bool(np.std(d["friction"]) > 0.05
                                                     and np.std(d["payload"]) > 0.3),
      "latency in the configured share of episodes": abs(lat - sum(cfg.dr.latency_probs)) < 0.15,
      "slopes within the configured maximum": bool(tilt.max() <= cfg.dr.slope_max_deg + 1e-3),
      "no pushes at level 0 (zero kick)": nominal["dv_push"] != nominal["dv_push"]
      or nominal["dv_push"] < 2 * nominal["dv_calm"] + 0.05,
      "pushes change base velocity at level 1": full["pushes"] > 0
      and full["dv_push"] > full["dv_calm"] + 0.1,
  }
  for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as f:
    json.dump({"runtime": rt.describe(),
               "nominal": {k: v for k, v in nominal.items() if k != "dr"},
               "randomised": {k: v for k, v in full.items() if k != "dr"},
               "checks": {k: bool(v) for k, v in checks.items()},
               "dr_config": dataclasses.asdict(cfg.dr)}, f, indent=2)
  if not all(checks.values()):
    sys.exit(1)
  print("Phase 3 randomisation check passed.")


if __name__ == "__main__":
  main()
