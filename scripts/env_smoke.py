"""Phase 1 smoke test for G1Env (~3 min on Kaggle 2xT4, mostly compilation).

Runs a random policy in many parallel envs with a short episode limit and fast
command resampling, then checks observations, termination, truncation,
auto-reset and command resampling. Saves a report, a trajectory of env 0, and
(if a GL backend is available) a video.

  python scripts/env_smoke.py                       # GPU, 512 envs
  python scripts/env_smoke.py --platform cpu --envs 8 --calls 2
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


def _cache_size(f):
  try:
    return f._cache_size()
  except Exception:  # private API; skip the check if it changes
    return None


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  p.add_argument("--envs", type=int, default=512)
  p.add_argument("--length", type=int, default=50, help="steps per jitted rollout call")
  p.add_argument("--calls", type=int, default=5)
  p.add_argument("--episode-length", type=int, default=60)
  p.add_argument("--action-std", type=float, default=0.3)
  p.add_argument("--out", default="outputs/phase1_smoke")
  p.add_argument("--no-video", action="store_true")
  args = p.parse_args()

  rt = device.setup(args.platform)
  import jax
  import jax.numpy as jnp
  import numpy as np

  from g1_rl import render
  from g1_rl.config import CommandConfig, EnvConfig
  from g1_rl.env import G1Env
  from g1_rl.vec_env import VecEnv

  print(f"runtime: {rt.describe()} | envs={args.envs}")
  cfg = EnvConfig(episode_length=args.episode_length,
                  command=CommandConfig(resample_steps=(20, 30)))
  env = G1Env(cfg)
  venv = VecEnv(env, rt.mesh, args.envs)

  def policy(obs, key):
    return args.action_std * jax.random.normal(key, (env.action_size,))

  def record(s):
    return dict(qpos=env.packer.unpack(s.data).qpos, step=s.info["step"], cmd=s.info["command"],
                done=s.done, trunc=s.truncation, obs_ok=jnp.isfinite(s.obs).all()
                & jnp.isfinite(s.priv_obs).all(), reward=s.reward)

  rollout = venv.make_rollout(policy, args.length, record)

  t0 = time.perf_counter()
  state = venv.reset(seed=0)
  jax.block_until_ready(state.obs)
  t_reset = time.perf_counter() - t0
  s0 = jax.device_get(state)

  keys = venv.keys(seed=1)
  recs, mets = [], []
  t0 = time.perf_counter()
  t_first = None
  for i in range(args.calls):
    state, keys, rec, met = rollout(state, keys)
    if i == 0:
      jax.block_until_ready(rec)
      t_first = time.perf_counter() - t0
      t1 = time.perf_counter()
    recs.append(jax.device_get(rec))
    mets.append(jax.device_get(met))
  t_run = time.perf_counter() - t1
  steps_timed = args.envs * args.length * max(args.calls - 1, 1)

  cat = lambda k: np.concatenate([r[k] for r in recs], axis=1)  # (N, T, ...)
  qpos, step, cmd = cat("qpos"), cat("step"), cat("cmd")
  done, trunc, obs_ok = cat("done"), cat("trunc"), cat("obs_ok")
  m = {k: float(sum(x[k].sum() for x in mets)) for k in mets[0]}
  n_ep = m["episode/count"]
  n_fell = m["episode/fell"]
  ep_len_mean = m["episode/length"] / max(n_ep, 1)

  # Command changes inside an episode (not caused by a reset).
  same_ep = (step[:, 1:] == step[:, :-1] + 1)
  cmd_changed = (np.abs(cmd[:, 1:] - cmd[:, :-1]).sum(-1) > 0) & same_ep

  grav0 = s0.obs[:, env.obs_layout["gravity"]]
  feet0 = s0.priv_obs[:, env.priv_layout["foot_height"]]
  after_done = step[:, 1:][done[:, :-1] > 0]

  checks = {
      "obs sizes are 100 / 139": (s0.obs.shape[1], s0.priv_obs.shape[1]) == (100, 139),
      "all observations finite": bool(obs_ok.all()),
      "reset: body upright (gravity_z < -0.9)": bool((grav0[:, 2] < -0.9).all()),
      "reset: lowest foot on the floor, other < 3 cm": bool(
          (np.abs(feet0.min(1)) < 0.005).all() and (feet0.max(1) < 0.03).all()),
      "episodes ended": n_ep > 0,
      "no episode longer than limit": bool((step <= args.episode_length).all()),
      "timeouts flagged as truncation": bool(trunc.sum() > 0 and (trunc <= done).all()),
      "falls detected": n_fell > 0,
      "auto-reset restarts step counter": bool(after_done.size == 0 or (after_done == 1).all()),
      "command resampled mid-episode": bool(cmd_changed.any()),
      "rollout compiled once (no retrace)": _cache_size(rollout) in (1, None),
  }
  report = dict(
      runtime=rt.describe(), n_envs=args.envs, reset_compile_s=round(t_reset, 1),
      rollout_compile_s=round(t_first, 1),
      env_steps_per_s=round(steps_timed / t_run) if args.calls > 1 else None,
      episodes=n_ep, falls=n_fell, timeouts=float(trunc.sum()), mean_episode_len=round(ep_len_mean, 1),
      mean_reward_per_step=float(cat("reward").mean()),
      checks=checks, env_config=json.loads(json.dumps(dataclasses.asdict(cfg), default=str)))

  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, "report.json"), "w") as f:
    json.dump(report, f, indent=2)

  print(f"compile: reset {t_reset:.0f}s, rollout {t_first:.0f}s | "
        f"throughput {report['env_steps_per_s']} env-steps/s (random policy)")
  print(f"episodes {n_ep:.0f} (falls {n_fell:.0f}, timeouts {trunc.sum():.0f}), "
        f"mean length {ep_len_mean:.1f} steps")
  for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

  if not args.no_video:
    traj = render.save_trajectory(os.path.join(args.out, "traj_env0.npz"), qpos[0],
                                  fps=1 / cfg.sim.ctrl_dt, xml_path=cfg.sim.xml_path)
    vid = render.try_render(traj, os.path.join(args.out, "traj_env0.mp4"))
    print(f"trajectory: {traj}" + (f" | video: {vid}" if vid else ""))

  if not all(checks.values()):
    sys.exit(1)
  print("Phase 1 smoke test passed.")


if __name__ == "__main__":
  main()
