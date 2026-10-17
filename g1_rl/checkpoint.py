"""Portable checkpoints: plain NumPy .npz + JSON. No framework needed to read them.

Two kinds of files are written into <run_dir>/:
  ckpt/ckpt_<iter>.npz (+ .json)  full TrainState for resuming training
  policy_latest.npz (+ .json), policy_final.npz (+ .json)
                                  just what is needed to run the policy: network
                                  weights, observation normaliser, action mapping
                                  and observation layout (see g1_rl/policy.py)
Writes go to a temp file first and are then renamed, so a run killed mid-save
never leaves a corrupt checkpoint behind.
"""

from __future__ import annotations

import glob
import json
import os
import re

import jax
import numpy as np


def _key_str(path) -> str:
  parts = []
  for p in path:
    for attr in ("key", "name", "idx"):
      if hasattr(p, attr):
        parts.append(str(getattr(p, attr)))
        break
    else:
      parts.append(str(p))
  return ".".join(parts)


def tree_to_flat(tree) -> dict:
  leaves = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))[0]
  return {_key_str(path): np.asarray(v) for path, v in leaves}


def flat_to_tree(template, flat: dict):
  paths, treedef = jax.tree_util.tree_flatten_with_path(template)
  leaves = []
  for path, ref in paths:
    k = _key_str(path)
    if k not in flat:
      raise KeyError(f"checkpoint is missing {k!r}")
    v = flat[k]
    if tuple(v.shape) != tuple(np.shape(ref)):
      raise ValueError(f"{k}: checkpoint shape {v.shape} != expected {np.shape(ref)} "
                       "(network sizes or observation layout changed?)")
    leaves.append(v.astype(np.asarray(ref).dtype))
  return jax.tree_util.tree_unflatten(treedef, leaves)


def _atomic_savez(path: str, arrays: dict, meta: dict) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  tmp = path + ".tmp.npz"
  np.savez(tmp, **arrays)
  os.replace(tmp, path)
  with open(path[:-4] + ".json.tmp", "w") as f:
    json.dump(meta, f, indent=2, default=str)
  os.replace(path[:-4] + ".json.tmp", path[:-4] + ".json")


# ------------------------------------------------------------------ training state
def save_checkpoint(run_dir: str, ts, meta: dict, keep: int = 3) -> str:
  it = int(jax.device_get(ts.iteration))
  path = os.path.join(run_dir, "ckpt", f"ckpt_{it:06d}.npz")
  _atomic_savez(path, tree_to_flat(ts), dict(meta, iteration=it))
  with open(os.path.join(run_dir, "ckpt", "latest.txt"), "w") as f:
    f.write(os.path.basename(path))
  for old in sorted(glob.glob(os.path.join(run_dir, "ckpt", "ckpt_*.npz")))[:-keep]:
    os.remove(old)
    if os.path.exists(old[:-4] + ".json"):
      os.remove(old[:-4] + ".json")
  return path


def find_checkpoint(path: str) -> str | None:
  """Accepts a ckpt .npz, a run dir, or a run dir's ckpt/ folder; returns the newest ckpt."""
  if path.endswith(".npz") and os.path.isfile(path):
    return path
  for d in (os.path.join(path, "ckpt"), path):
    latest = os.path.join(d, "latest.txt")
    if os.path.isfile(latest):
      p = os.path.join(d, open(latest).read().strip())
      if os.path.isfile(p):
        return p
    found = sorted(glob.glob(os.path.join(d, "ckpt_*.npz")))
    if found:
      return found[-1]
  return None


def load_checkpoint(path: str, template):
  ckpt = find_checkpoint(path)
  if ckpt is None:
    raise FileNotFoundError(f"no checkpoint found at {path}")
  with np.load(ckpt) as z:
    flat = {k: z[k] for k in z.files}
  meta = {}
  if os.path.exists(ckpt[:-4] + ".json"):
    meta = json.load(open(ckpt[:-4] + ".json"))
  return flat_to_tree(template, flat), meta, ckpt


# ------------------------------------------------------------------ policy export
def export_policy(path: str, ts, env, ppo_cfg, extra_meta: dict | None = None) -> str:
  """Everything needed to run the trained policy, independent of the training code."""
  params = jax.device_get(ts.params["policy"])
  arrays = {f"layer{i}.w": np.asarray(l["w"]) for i, l in enumerate(params["net"])}
  arrays.update({f"layer{i}.b": np.asarray(l["b"]) for i, l in enumerate(params["net"])})
  arrays["log_std"] = np.asarray(params["log_std"])
  stats = jax.device_get(ts.obs_stats)
  arrays["obs_mean"], arrays["obs_var"] = np.asarray(stats.mean), np.asarray(stats.var)
  r = env.robot
  arrays["home_ctrl"] = np.asarray(r.home_ctrl, np.float32)
  arrays["ctrl_lo"], arrays["ctrl_hi"] = np.asarray(r.ctrl_lo), np.asarray(r.ctrl_hi)
  arrays["kp"] = env.dr.kp0
  arrays["kd"] = env.dr.kd0
  layout = {k: [v.start, v.stop] for k, v in env.obs_layout.items() if k != "_size"}
  meta = {
      "format": "g1_rl.policy.v1",
      "num_layers": len(params["net"]),
      "activation": ppo_cfg.activation,
      "obs_clip": ppo_cfg.obs_clip,
      "obs_size": env.obs_size,
      "frame_size": int(getattr(env, "frame_size", env.obs_size)),  # v0.9: obs = history x frame
      "history": int(getattr(env, "history", 1)),
      "action_size": env.action_size,
      "obs_layout": layout,
      "action_scale": [float(x) for x in env.action_scale],  # per joint (a scalar before v0.8)
      "ctrl_dt": env.dt,
      "joint_names": list(r.actuator_names),
      "gait_freq_range": list(env.cfg.gait_freq),
      "iteration": int(jax.device_get(ts.iteration)),
      "action_mapping": "target = clip(home_ctrl + action_scale * clip(action, -1, 1), ctrl_lo, ctrl_hi)",
      "observation": ("frame = [gyro(3), gravity_body(3), command(3), joint_pos-home(29), joint_vel(29), "
                      "last_action(29), cos(phase)(2), sin(phase)(2)]; joint order = joint_names; "
                      "obs = the last `history` frames concatenated, newest first; obs_layout = one frame"),
  }
  meta.update(extra_meta or {})
  _atomic_savez(path, arrays, meta)
  return path


def iteration_of(path: str) -> int:
  m = re.search(r"ckpt_(\d+)\.npz$", path)
  return int(m.group(1)) if m else -1
