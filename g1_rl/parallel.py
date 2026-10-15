"""Multi-GPU data parallelism over a 1-D device mesh with axis "batch".

Default mode is "gspmd": a plain ``jax.jit`` whose inputs/outputs are sharded
along the leading (env) axis. Every robot is independent, so XLA partitions the
vmapped simulation across devices; PPO's mean-over-batch loss automatically
becomes an all-reduce of gradients. This is also what you get on 1 GPU.

The only collective in the simulator (see ``count_collectives``) is a 1-byte
all-reduce of the constraint solver's batched while-loop "any env still
iterating?" flag. It costs a few microseconds per solver iteration.

"shard_map" mode is kept as an option, but shard_map/pmap around MJX's forward
pass segfaults XLA in multi-device runs (reproduced on jax 0.11.2 / mujoco
3.16.0, CPU multi-device), so it is not the default.
"""

from __future__ import annotations

import re

import jax
from jax.sharding import NamedSharding, PartitionSpec as P

AXIS = "batch"
BATCH = P(AXIS)
REPLICATED = P()


def batch_sharding(mesh) -> NamedSharding:
  return NamedSharding(mesh, BATCH)


def replicated_sharding(mesh) -> NamedSharding:
  return NamedSharding(mesh, REPLICATED)


def jit_batched(fn, mesh, donate_argnums=(), static_argnums=()):
  """jit `fn` with every input and output leaf sharded on its leading axis."""
  s = batch_sharding(mesh)
  return jax.jit(fn, in_shardings=s, out_shardings=s,
                 donate_argnums=donate_argnums, static_argnums=static_argnums)


def put_batched(tree, mesh):
  return jax.device_put(tree, batch_sharding(mesh))


def put_replicated(tree, mesh):
  return jax.device_put(tree, replicated_sharding(mesh))


def check_divisible(n: int, mesh) -> None:
  nd = mesh.devices.size
  if n % nd:
    raise ValueError(f"num_envs={n} must be divisible by the number of devices ({nd}).")


_COLLECTIVE_RE = re.compile(
    r"\b(all-gather|all-reduce|all-to-all|collective-permute|reduce-scatter)(-start)?\(")


def count_collectives(jitted_fn, *args) -> dict:
  """Count cross-device ops in the compiled HLO (should be 0 for the simulator)."""
  return count_collectives_text(jitted_fn.lower(*args).compile().as_text())


def count_collectives_text(hlo: str) -> dict:
  counts: dict = {}
  for m in _COLLECTIVE_RE.finditer(hlo):
    counts[m.group(1)] = counts.get(m.group(1), 0) + 1
  return counts


# Optional explicit-SPMD mode ---------------------------------------------------
try:  # jax >= 0.6
  _shard_map = jax.shard_map
  _NEW_API = True
except AttributeError:  # older jax
  from jax.experimental.shard_map import shard_map as _shard_map  # type: ignore
  _NEW_API = False


def shard_map(fn, mesh, in_specs, out_specs, check: bool = False):
  if _NEW_API:
    return _shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check)
  return _shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=check)
