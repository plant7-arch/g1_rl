"""Phase 2 check: reward terms inside the real simulator (~3 min on Kaggle 2xT4).

Runs a random policy (the same kind of behaviour PPO starts from) and prints each
term's average weighted contribution per step, so you can see the reward balance
before spending a night on training. Also checks signs, finiteness, and that the
termination term matches the number of falls.

  python scripts/reward_check.py
  python scripts/reward_check.py --platform cpu --envs 8 --calls 2 --length 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  p.add_argument("--envs", type=int, default=1024)
  p.add_argument("--length", type=int, default=100)
  p.add_argument("--calls", type=int, default=3)
  p.add_argument("--action-std", type=float, default=0.3)
  p.add_argument("--out", default="outputs/phase2_rewards.json")
  args = p.parse_args()

  rt = device.setup(args.platform)
  import jax
  import numpy as np

  from g1_rl.config import EnvConfig
  from g1_rl.env import G1Env
  from g1_rl.vec_env import VecEnv

  cfg = EnvConfig()
  env = G1Env(cfg)
  venv = VecEnv(env, rt.mesh, args.envs)
  print(f"runtime: {rt.describe()} | envs={args.envs} | random policy std={args.action_std}")

  def policy(obs, key):
    return args.action_std * jax.random.normal(key, (env.action_size,))

  rollout = venv.make_rollout(policy, args.length)
  state, keys = venv.reset(0), venv.keys(1)
  sums = None
  for _ in range(args.calls):
    state, keys, rew, met = rollout(state, keys)
    met = jax.device_get(met)
    rew_sum = float(np.asarray(jax.device_get(rew)).sum())
    part = {k: float(np.asarray(v).sum()) for k, v in met.items()}
    part["_reward"] = rew_sum
    sums = part if sums is None else {k: sums[k] + part[k] for k in sums}

  steps = args.envs * args.length * args.calls
  W = cfg.reward.weights
  terms = {k[len("reward/"):]: v / steps for k, v in sums.items() if k.startswith("reward/")}
  pos = sum(v for v in terms.values() if v > 0)
  neg = sum(v for v in terms.values() if v < 0)
  n_ep, n_fell = sums["episode/count"], sums["episode/fell"]

  print(f"\n{'term':<18}{'weight':>10}{'mean/step':>12}{'share':>8}")
  for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1])):
    share = v / (pos if v > 0 else -neg) if v else 0.0
    print(f"{k:<18}{W[k]:>10.3g}{v:>12.5f}{share:>8.0%}")
  print(f"{'positive total':<28}{pos:>12.5f}\n{'negative total':<28}{neg:>12.5f}")
  print(f"{'reward (after clip)':<28}{sums['_reward'] / steps:>12.5f}  "
        f"| clipped on {sums['debug/reward_clipped'] / steps:.0%} of steps")
  print(f"episodes ended {n_ep:.0f} (falls {n_fell:.0f}), "
        f"mean length {sums['episode/length'] / max(n_ep, 1):.0f} steps")

  checks = {
      "all terms finite": all(np.isfinite(v) for v in terms.values()),
      "task rewards positive": all(terms[k] > 0 for k in ("tracking_lin_vel", "tracking_ang_vel")),
      "penalty terms <= 0": all(v <= 1e-12 for k, v in terms.items() if W[k] < 0),
      "termination matches falls": np.isclose(terms.get("termination", 0) * steps,
                                              n_fell * W["termination"] * cfg.sim.ctrl_dt, rtol=1e-3),
      "no single penalty larger than all rewards": all(-v < pos for v in terms.values() if v < 0),
  }
  for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as f:
    json.dump(dict(runtime=rt.describe(), steps=steps, terms_per_step=terms, positive=pos,
                   negative=neg, checks={k: bool(v) for k, v in checks.items()}), f, indent=2)
  if not all(checks.values()):
    sys.exit(1)
  print("Phase 2 reward check passed.")


if __name__ == "__main__":
  main()
