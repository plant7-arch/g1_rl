"""Adam with global-norm gradient clipping, in plain JAX (no optax)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class AdamState(NamedTuple):
  mu: dict
  nu: dict
  count: jax.Array


def adam_init(params) -> AdamState:
  zeros = jax.tree.map(jnp.zeros_like, params)
  return AdamState(mu=zeros, nu=jax.tree.map(jnp.zeros_like, params),
                   count=jnp.zeros((), jnp.int32))


def global_norm(tree):
  return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(tree)))


def clip_by_global_norm(grads, max_norm):
  norm = global_norm(grads)
  scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
  return jax.tree.map(lambda g: g * scale, grads), norm


def adam_update(params, grads, state: AdamState, lr, b1=0.9, b2=0.999, eps=1e-8):
  count = state.count + 1
  mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, grads)
  nu = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, grads)
  c = count.astype(jnp.float32)
  mhat_scale, vhat_scale = 1.0 / (1 - b1 ** c), 1.0 / (1 - b2 ** c)
  new = jax.tree.map(lambda p, m, v: p - lr * (m * mhat_scale) / (jnp.sqrt(v * vhat_scale) + eps),
                     params, mu, nu)
  return new, AdamState(mu, nu, count)
