"""Running mean / variance observation normalizer (merged batch statistics)."""

from __future__ import annotations

from typing import NamedTuple

import math

import jax
import jax.numpy as jnp


def np_prod(shape):
  return math.prod(shape)


class RunningStats(NamedTuple):
  mean: jax.Array
  var: jax.Array
  count: jax.Array


def init(size: int) -> RunningStats:
  return RunningStats(jnp.zeros((size,), jnp.float32), jnp.ones((size,), jnp.float32),
                      jnp.asarray(1e-4, jnp.float32))


def update(stats: RunningStats, x) -> RunningStats:
  """Merge a batch `x` of shape (..., size) into the running statistics.

  Reduces over all leading axes directly (no reshape): flattening a (T, N, size)
  rollout sharded on N would interleave devices and force XLA to gather it.
  """
  axes = tuple(range(x.ndim - 1))
  n = jnp.asarray(np_prod(x.shape[:-1]), jnp.float32)
  b_mean = jnp.mean(x, axis=axes)
  b_var = jnp.mean(jnp.square(x - b_mean), axis=axes)
  total = stats.count + n
  delta = b_mean - stats.mean
  mean = stats.mean + delta * (n / total)
  m2 = stats.var * stats.count + b_var * n + jnp.square(delta) * stats.count * n / total
  return RunningStats(mean, m2 / total, total)


def normalize(stats: RunningStats, x, clip: float = 10.0):
  return jnp.clip((x - stats.mean) / jnp.sqrt(stats.var + 1e-8), -clip, clip)
