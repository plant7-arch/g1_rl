"""Thin, jit/vmap-friendly wrappers around MJX shared by the env and the benchmark."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from mujoco import mjx


class DataPacker:
  """Converts mjx.Data <-> a tuple of its non-empty leaves.

  mjx.Data contains zero-size arrays (e.g. act, mocap_pos, ten_J for this robot).
  Passing batched zero-size arrays across a shard_map / multi-device jit boundary
  segfaults XLA (verified on jax 0.11 / mujoco 3.16), so everything that crosses
  a jit boundary is packed, and unpacked again inside the traced function.
  """

  def __init__(self, mjx_model):
    template = jax.eval_shape(lambda: mjx.make_data(mjx_model))
    leaves, self.treedef = jax.tree.flatten(template)
    self.empty = [0 in l.shape for l in leaves]
    self.specs = [(l.shape, l.dtype) for l in leaves]

  def pack(self, data) -> tuple:
    leaves = jax.tree.leaves(data)
    assert len(leaves) == len(self.empty), "Data structure changed"
    return tuple(l for l, e in zip(leaves, self.empty) if not e)

  def unpack(self, packed: tuple):
    it = iter(packed)
    leaves = [jnp.zeros(s, d) if e else next(it) for e, (s, d) in zip(self.empty, self.specs)]
    return jax.tree.unflatten(self.treedef, leaves)


def init_data(mjx_model, qpos, ctrl):
  """Single (unbatched) mjx.Data at a given configuration, with kinematics computed."""
  d = mjx.make_data(mjx_model)
  d = d.replace(qpos=qpos, qvel=jnp.zeros_like(d.qvel), ctrl=ctrl)
  return mjx.forward(mjx_model, d)


def step_ctrl(mjx_model, data, ctrl, n_substeps: int):
  """Apply position targets and advance n_substeps physics steps (one control step)."""
  data = data.replace(ctrl=ctrl)

  def body(d, _):
    return mjx.step(mjx_model, d), None

  data, _ = lax.scan(body, data, None, length=n_substeps)
  return data


def tree_where(cond, on_true, on_false):
  """Per-env select between two Data pytrees (cond is a scalar bool inside vmap)."""
  return jax.tree.map(lambda a, b: jnp.where(cond, a, b), on_true, on_false)


def is_unhealthy(data, min_height: float = 0.45):
  bad_num = ~(jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel)))
  return (data.qpos[2] < min_height) | bad_num
