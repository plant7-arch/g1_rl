"""Batched, multi-GPU wrapper around G1Env.

Every leaf of the batched State has a leading env axis that is split across the
devices of `mesh`. PPO (Phase 4) builds its own fused rollout+update jit on top
of `env.reset` / `env.step`; this wrapper is for smoke tests and evaluation.
"""

from __future__ import annotations

import jax

from g1_rl import parallel
from g1_rl.env import G1Env


class VecEnv:

  def __init__(self, env: G1Env, mesh, num_envs: int):
    parallel.check_divisible(num_envs, mesh)
    self.env, self.mesh, self.num_envs = env, mesh, num_envs
    self.reset_fn = parallel.jit_batched(jax.vmap(env.reset), mesh)
    self.step_fn = parallel.jit_batched(jax.vmap(env.step), mesh, donate_argnums=(0,))

  def keys(self, seed: int):
    """Per-env PRNG keys, already sharded (unsharded inputs would trigger a recompile)."""
    return parallel.put_batched(jax.random.split(jax.random.PRNGKey(seed), self.num_envs), self.mesh)

  def reset(self, seed: int = 0):
    return self.reset_fn(self.keys(seed))

  def step(self, state, actions):
    return self.step_fn(state, actions)

  def make_rollout(self, policy_fn, length: int, record=lambda s: s.reward):
    """Jitted `length`-step rollout under `policy_fn(obs, key) -> action` (per env).

    Returns (final_state, per-step records stacked on axis 1, metrics summed over time).
    `record(state)` picks what to keep per step (default: reward).
    """
    env = self.env

    def one(state, key):
      def body(carry, _):
        s, k = carry
        k, ka = jax.random.split(k)
        s = env.step(s, policy_fn(s.obs, ka))
        return (s, k), (record(s), s.metrics)

      (state, key), (rec, metrics) = jax.lax.scan(body, (state, key), None, length=length)
      return state, key, rec, jax.tree.map(lambda x: x.sum(0), metrics)

    return parallel.jit_batched(jax.vmap(one), self.mesh, donate_argnums=(0,))
