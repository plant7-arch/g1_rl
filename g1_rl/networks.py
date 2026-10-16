"""MLPs and the diagonal-Gaussian policy, in plain JAX (params are nested dicts/lists).

Policy: mean = MLP(normalized obs); log_std is a learned, state-independent vector
(as in rsl_rl). Actions are sampled from N(mean, std) and clipped to [-1, 1] by the
env; log-probabilities are of the unclipped sample. At deployment the mean is used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

ACTIVATIONS = {"elu": jax.nn.elu, "silu": jax.nn.silu, "tanh": jnp.tanh, "relu": jax.nn.relu}
LOG_STD_MIN, LOG_STD_MAX = np.log(0.02), np.log(2.0)
_LOG_2PI = float(np.log(2 * np.pi))


def init_mlp(key, sizes, last_scale=1.0) -> list:
  """Orthogonal init (gain sqrt(2) for hidden layers, `last_scale` for the output layer)."""
  keys = jax.random.split(key, len(sizes) - 1)
  layers = []
  for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
    gain = np.sqrt(2.0) if i < len(sizes) - 2 else last_scale
    w = jax.nn.initializers.orthogonal(gain)(k, (a, b), jnp.float32)
    layers.append({"w": w, "b": jnp.zeros((b,), jnp.float32)})
  return layers


def mlp(layers, x, activation="elu"):
  act = ACTIVATIONS[activation]
  for layer in layers[:-1]:
    x = act(x @ layer["w"] + layer["b"])
  return x @ layers[-1]["w"] + layers[-1]["b"]


def init_actor_critic(key, obs_size, priv_size, action_size, policy_hidden, value_hidden,
                      init_std) -> dict:
  kp, kv = jax.random.split(key)
  return {
      "policy": {
          "net": init_mlp(kp, (obs_size, *policy_hidden, action_size), last_scale=0.01),
          "log_std": jnp.full((action_size,), np.log(init_std), jnp.float32),
      },
      "value": {"net": init_mlp(kv, (priv_size, *value_hidden, 1), last_scale=1.0)},
  }


def policy_dist(policy_params, obs_norm, activation="elu"):
  mean = mlp(policy_params["net"], obs_norm, activation)
  log_std = jnp.clip(policy_params["log_std"], LOG_STD_MIN, LOG_STD_MAX)
  return mean, jnp.broadcast_to(log_std, mean.shape)


def value(value_params, priv_norm, activation="elu"):
  return mlp(value_params["net"], priv_norm, activation)[..., 0]


def log_prob(mean, log_std, x):
  z = (x - mean) * jnp.exp(-log_std)
  return jnp.sum(-0.5 * z * z - log_std - 0.5 * _LOG_2PI, axis=-1)


def entropy(log_std):
  return jnp.sum(log_std + 0.5 * (1.0 + _LOG_2PI), axis=-1)


def kl_divergence(mean_p, log_std_p, mean_q, log_std_q):
  """KL(p || q) for diagonal Gaussians, summed over action dims."""
  var_p, var_q = jnp.exp(2 * log_std_p), jnp.exp(2 * log_std_q)
  return jnp.sum(log_std_q - log_std_p + (var_p + jnp.square(mean_p - mean_q)) / (2 * var_q) - 0.5,
                 axis=-1)
