"""Logger throughput / ETA must not be fooled by irregular log-call timing."""

import json

from g1_rl.logger import Logger


def test_sps_is_average_since_first_log(tmp_path, capsys):
  log = Logger(str(tmp_path), total_iters=100, batch_size=1000, log_every=1)
  t = [0.0]
  log.clock = lambda: t[0]
  m = {"episode/return": 1.0}
  log.log(1, m)              # first iteration (after compile): reference point
  for it, now in [(2, 1.0), (3, 2.0), (4, 3.0), (5, 3.01)]:  # 5 logged right after 4 (checkpoint)
    t[0] = now
    log.log(it, m)
  log.close()
  rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
  assert rows[0]["sps"] is None
  assert abs(rows[3]["sps"] - 1000) < 0.1           # 3 iterations in 3 s
  assert abs(rows[4]["sps"] - 4000 / 3.01) < 0.1    # not inflated by the 10 ms gap
