"""Train the G1 walking policy with PPO.

  python scripts/train.py --preset smoke                    # < 5 min check (no video)
  python scripts/train.py --preset full --max-hours 11.5    # overnight Kaggle run
  python scripts/train.py --preset full --resume outputs/runs/full_xxx   # continue a run

Outputs (in --out-dir/<run-name>/):
  metrics.jsonl            every iteration's metrics (plot with scripts/plot_metrics.py)
  ckpt/ckpt_<iter>.npz     full training state for --resume (last 3 kept)
  policy_latest.npz/.json  policy for running/evaluating (updated at every checkpoint)
  policy_final.npz/.json   policy after the last iteration
  video_walk_forward.mp4   10 s of the final policy walking forward (full runs)
  config.json, summary.json
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--preset", default="full", choices=["full", "smoke"])
  p.add_argument("--platform", default="auto", choices=["auto", "gpu", "cpu"])
  p.add_argument("--run-name", default=None, help="default: <preset>_<date-time>")
  p.add_argument("--out-dir", default="outputs/runs")
  p.add_argument("--resume", default=None, help="run dir or ckpt .npz to continue from")
  p.add_argument("--init-policy", default=None,
                 help="warm start from an exported policy .npz (e.g. g1_walk_v4/policy_final.npz)")
  p.add_argument("--critic-warmup", type=int, default=None, help="iterations training only the critic")
  p.add_argument("--dr-level-start", type=float, default=None)
  p.add_argument("--max-hours", type=float, default=None,
                 help="stop cleanly (checkpoint saved) before this much time; Kaggle: 11.5")
  p.add_argument("--total-steps", type=float, default=None, help="override total env steps")
  p.add_argument("--num-envs", type=int, default=None)
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--stop-after", type=int, default=None, help="stop after N iterations this session")
  p.add_argument("--video", dest="video", action="store_true", default=None)
  p.add_argument("--no-video", dest="video", action="store_false")
  args = p.parse_args()

  rt = device.setup(args.platform)
  from g1_rl.config import PRESETS
  from g1_rl.trainer import train

  cfg = PRESETS[args.preset]
  over = {k: v for k, v in [("total_env_steps", args.total_steps), ("num_envs", args.num_envs),
                            ("seed", args.seed), ("critic_warmup_iters", args.critic_warmup),
                            ("dr_level_start", args.dr_level_start)] if v is not None}
  cfg = dataclasses.replace(cfg, **over)
  name = args.run_name or f"{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}"
  video = args.video if args.video is not None else args.preset == "full"
  summary = train(cfg, os.path.join(args.out_dir, name), rt, resume=args.resume,
                  max_hours=args.max_hours, video=video, stop_after=args.stop_after,
                  init_policy=args.init_policy)
  sys.exit(0 if summary["stopped"] != "diverged" else 1)


if __name__ == "__main__":
  main()
