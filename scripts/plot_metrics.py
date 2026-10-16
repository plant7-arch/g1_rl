"""Plot a training run's metrics.jsonl into one PNG (morning check of an overnight run).

  python scripts/plot_metrics.py outputs/runs/<run>            # -> <run>/training_curves.png
  python scripts/plot_metrics.py outputs/runs/<run> --show-terms   # extra page with reward terms
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [
    ("Episode return", ["episode/return"]),
    ("Episode length (steps, max 1000)", ["episode/length"]),
    ("Fraction of episodes ending in a fall", ["episode/fall_frac"]),
    ("Velocity tracking (fraction of max)", ["curriculum/tracking", "curriculum/tracking_yaw"]),
    ("Randomisation level (curriculum)", ["curriculum/dr_level"]),
    ("Throughput (env steps / s)", ["sps"]),
    ("KL and learning rate", ["train/kl", "train/lr"]),
    ("Policy std / value explained var.", ["policy/std", "value/explained_var"]),
    ("Losses", ["loss/policy", "loss/value"]),
]


def load(run_dir):
  rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
  rows.sort(key=lambda r: r["iteration"])
  dedup = {r["iteration"]: r for r in rows}  # a resumed run may repeat iterations
  return list(dedup.values())


def series(rows, key):
  x = np.array([r["env_steps"] / 1e6 for r in rows])
  y = np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)
  return x, y


def smooth(y, k=9):
  if len(y) < k:
    return y
  pad = np.pad(y, (k // 2, k // 2), mode="edge")
  ok = np.isfinite(pad)
  num = np.convolve(np.where(ok, pad, 0), np.ones(k), "valid")
  den = np.convolve(ok.astype(float), np.ones(k), "valid")
  return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run_dir")
  p.add_argument("--show-terms", action="store_true")
  a = p.parse_args()
  rows = load(a.run_dir)
  fig, axes = plt.subplots(3, 3, figsize=(15, 10))
  for ax, (title, keys) in zip(axes.ravel(), PANELS):
    for k in keys:
      x, y = series(rows, k)
      if np.isfinite(y).any():
        ax.plot(x, y, alpha=0.25, lw=0.8)
        ax.plot(x, smooth(y), lw=1.6, label=k.split("/")[-1], color=ax.lines[-1].get_color())
    if k in ("train/lr",) or title.startswith("KL"):
      ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("env steps (M)", fontsize=8)
    ax.grid(alpha=0.3)
    if len(keys) > 1:
      ax.legend(fontsize=8)
  last = rows[-1]
  fig.suptitle(f"{os.path.basename(os.path.normpath(a.run_dir))}: iteration {last['iteration']}, "
               f"{last['env_steps'] / 1e6:.0f}M steps, {last['wall_s'] / 3600:.1f} h", fontsize=12)
  fig.tight_layout()
  out = os.path.join(a.run_dir, "training_curves.png")
  fig.savefig(out, dpi=110)
  print(f"wrote {out}")

  if a.show_terms:
    terms = sorted(k for k in last if k.startswith("reward/"))
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in terms:
      x, y = series(rows, k)
      ax.plot(x, smooth(y), label=k[len("reward/"):])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Reward terms (weighted, per step)")
    ax.set_xlabel("env steps (M)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    out2 = os.path.join(a.run_dir, "reward_terms.png")
    fig.tight_layout()
    fig.savefig(out2, dpi=110)
    print(f"wrote {out2}")


if __name__ == "__main__":
  main()
