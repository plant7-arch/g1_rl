"""Compact training log: one console line every N iterations, every iteration to a JSONL file."""

from __future__ import annotations

import json
import math
import os
import time


def _fmt_time(s: float) -> str:
  if not math.isfinite(s) or s < 0:
    return "?"
  h, m = divmod(int(s) // 60, 60)
  return f"{h}h{m:02d}m" if h else f"{m}m{int(s) % 60:02d}s"


class Logger:

  def __init__(self, run_dir: str, total_iters: int, batch_size: int, log_every: int,
               start_iter: int = 0):
    os.makedirs(run_dir, exist_ok=True)
    self.path = os.path.join(run_dir, "metrics.jsonl")
    self.file = open(self.path, "a")
    self.total, self.batch, self.every = total_iters, batch_size, log_every
    self.start_iter = start_iter
    self.t0 = time.time()
    self.t_last, self.it_last = None, None
    self.sps = float("nan")

  def log(self, it: int, m: dict) -> None:
    """`it` = number of completed iterations (1-based)."""
    now = time.time()
    if self.t_last is not None and it > self.it_last:
      inst = (it - self.it_last) * self.batch / (now - self.t_last)
      self.sps = inst if not math.isfinite(self.sps) else 0.8 * self.sps + 0.2 * inst
    self.t_last, self.it_last = now, it
    rec = {"iteration": it, "env_steps": it * self.batch, "wall_s": round(now - self.t0, 1),
           "sps": round(self.sps, 1) if math.isfinite(self.sps) else None}
    rec.update({k: (float(v) if math.isfinite(float(v)) else None) for k, v in m.items()})
    self.file.write(json.dumps(rec) + "\n")
    self.file.flush()
    if it % self.every == 0 or it == self.total or it == self.start_iter + 1:
      eta = (self.total - it) * self.batch / self.sps if math.isfinite(self.sps) else float("nan")
      g = lambda k: float(m.get(k, float("nan")))

      def f(k, spec, width):
        v = g(k)
        return format(v, spec) if math.isfinite(v) else "n/a".rjust(width)

      sps = f"{self.sps / 1e3:5.1f}k" if math.isfinite(self.sps) else "    - "
      print(f"[{it:5d}/{self.total} {it * self.batch / 1e6:6.1f}M] {sps} sps | "
            f"ret {f('episode/return', '7.2f', 7)} len {f('episode/length', '5.0f', 5)} "
            f"fall {f('episode/fall_frac', '4.2f', 4)} | track {g('curriculum/tracking'):4.2f} "
            f"yaw {g('curriculum/tracking_yaw'):4.2f} | dr {g('curriculum/dr_level'):4.2f} | "
            f"kl {g('train/kl'):.4f} lr {g('train/lr'):.1e} std {g('policy/std'):.2f} "
            f"ev {g('value/explained_var'):4.2f} | {_fmt_time(now - self.t0)} ETA {_fmt_time(eta)}",
            flush=True)

  def close(self):
    self.file.close()
