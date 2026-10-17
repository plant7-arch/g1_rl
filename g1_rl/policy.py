"""Load and run an exported policy (policy_*.npz) on any machine with jax + numpy.

    from g1_rl.policy import Policy
    pi = Policy.load("outputs/runs/<run>/policy_final.npz")      # GPU if available, else CPU
    action = pi(obs)                  # obs: (100,) or (B, 100) -> action in [-1, 1]
    targets = pi.joint_targets(obs)   # PD position targets (rad) for the 29 joints

Uses the deterministic mean action (no exploration noise). Observation layout and
the action mapping are stored in the file's .json next to it (pi.meta).
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np


class Policy:

  def __init__(self, arrays: dict, meta: dict, device=None):
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from g1_rl import networks  # noqa: PLC0415

    self.meta = meta
    self.device = device or _pick_device()
    put = lambda x: jax.device_put(jnp.asarray(x), self.device)
    n = meta["num_layers"]
    self._layers = [{"w": put(arrays[f"layer{i}.w"]), "b": put(arrays[f"layer{i}.b"])} for i in range(n)]
    self._mean, self._var = put(arrays["obs_mean"]), put(arrays["obs_var"])
    self.home = np.asarray(arrays["home_ctrl"])
    self.lo, self.hi = np.asarray(arrays["ctrl_lo"]), np.asarray(arrays["ctrl_hi"])
    self.kp, self.kd = np.asarray(arrays["kp"]), np.asarray(arrays["kd"])
    self.action_scale = np.asarray(meta["action_scale"], np.float64)  # per joint, or scalar (< v0.8)
    act, clip = meta["activation"], float(meta["obs_clip"])

    def forward(layers, mean, var, obs):
      x = jnp.clip((obs - mean) / jnp.sqrt(var + 1e-8), -clip, clip)
      return jnp.clip(networks.mlp(layers, x, act), -1.0, 1.0)

    self._forward = jax.jit(forward)
    # Host (NumPy) copies for building traceable functions (see jax_fn).
    self._np = ([{k: np.asarray(v) for k, v in l.items()} for l in self._layers],
                np.asarray(self._mean), np.asarray(self._var))
    self._raw_forward = forward

  def jax_fn(self):
    """Pure obs -> action function for use inside other jitted code (e.g. a batched
    MJX rollout sharded over several GPUs). Weights are embedded as host constants,
    so the function is not tied to the device the policy was loaded on."""
    layers, mean, var = self._np
    return lambda obs: self._raw_forward(layers, mean, var, obs)

  @classmethod
  def load(cls, path: str, device=None) -> "Policy":
    with np.load(path) as z:
      arrays = {k: z[k] for k in z.files}
    meta_path = path[:-4] + ".json"
    if not os.path.exists(meta_path):
      raise FileNotFoundError(f"policy metadata {meta_path} not found next to {path}")
    return cls(arrays, json.load(open(meta_path)), device)

  def __call__(self, obs):
    return self._forward(self._layers, self._mean, self._var, obs)

  def joint_targets(self, obs):
    a = np.asarray(self(obs))
    return np.clip(self.home + self.action_scale * a, self.lo, self.hi)


def _pick_device():
  import jax  # noqa: PLC0415
  try:
    gpus = jax.devices("gpu")
  except RuntimeError:
    gpus = []
  if gpus:
    return gpus[0]
  warnings.warn("No GPU visible to JAX; running the policy on CPU.")
  return jax.devices("cpu")[0]
