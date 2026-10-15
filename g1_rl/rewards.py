"""Reward terms. Phase 1 placeholder: only an alive bonus so the env runs end to end.

Phase 2 replaces `compute` with the full set (velocity tracking, gait shaping,
regularisation penalties). The env passes everything a term could need in `ctx`.
"""

import jax.numpy as jnp


def compute(ctx: dict, cfg=None) -> dict:
  return {"alive": jnp.ones(())}


def total(terms: dict, ctrl_dt: float) -> jnp.ndarray:
  return sum(terms.values()) * ctrl_dt
