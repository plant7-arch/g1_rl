"""Proximal Policy Optimization, written from scratch in JAX.

Structure follows the well-tested designs of brax and rsl_rl (asymmetric
actor-critic, GAE, clipped surrogate and value losses, per-minibatch advantage
normalisation, KL-adaptive learning rate, running observation normalisation),
but every piece is implemented here so there is no dependency on them.

One call of the jitted `train_step` does a whole PPO iteration on the GPUs:
  rollout (num_envs x unroll_length env steps) -> GAE -> num_epochs x num_minibatches
  gradient steps -> normaliser + curriculum update -> scalar metrics.
The only host traffic is the small metrics dict (for logging).

Multi-GPU: envs are split across devices (batch sharding). Minibatches are
drawn *within* each device's block of samples, so shuffling needs no
cross-device traffic; gradients of the global mean loss become an all-reduce.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from g1_rl import networks as nets
from g1_rl import normalizer as norm
from g1_rl import optim, symmetry
from g1_rl.config import PPOConfig
from g1_rl.parallel import AXIS


class TrainState(NamedTuple):
  params: dict
  opt: optim.AdamState
  obs_stats: norm.RunningStats
  priv_stats: norm.RunningStats
  lr: jax.Array
  dr_level: jax.Array
  key: jax.Array
  iteration: jax.Array


def init_train_state(key, obs_size, priv_size, action_size, cfg: PPOConfig) -> TrainState:
  k_params, k_state = jax.random.split(key)
  params = nets.init_actor_critic(k_params, obs_size, priv_size, action_size,
                                  cfg.policy_hidden, cfg.value_hidden, cfg.init_std)
  return TrainState(
      params=params, opt=optim.adam_init(params),
      obs_stats=norm.init(obs_size), priv_stats=norm.init(priv_size),
      lr=jnp.asarray(cfg.learning_rate, jnp.float32),
      dr_level=jnp.asarray(cfg.dr_level_start, jnp.float32),
      key=k_state, iteration=jnp.zeros((), jnp.int32))


# ------------------------------------------------------------------------- GAE
def compute_gae(rewards, values, dones, truncations, gamma, lam):
  """Generalised advantage estimation over a (T, N) rollout.

  values has T+1 rows: V(s_t) for each step plus V of the observation after the
  last step. A step with done=1 does not bootstrap from the next observation
  (which belongs to a new episode). If the episode ended by the time limit
  (truncation=1) it was not a failure, so the missing future is approximated by
  V(s_t) (rsl_rl's time-out bootstrap).
  """
  rewards = rewards + gamma * values[:-1] * truncations
  not_done = 1.0 - dones
  deltas = rewards + gamma * values[1:] * not_done - values[:-1]

  def body(next_adv, x):
    delta, nd = x
    adv = delta + gamma * lam * nd * next_adv
    return adv, adv

  _, adv = lax.scan(body, jnp.zeros_like(deltas[0]), (deltas, not_done), reverse=True)
  return adv, adv + values[:-1]


# ------------------------------------------------------------------------ loss
def ppo_loss(params, mb, obs_stats, priv_stats, old_log_std, cfg: PPOConfig, mirror=None,
             train_policy=True):
  """`mirror` = (obs_perm, obs_sign, act_perm, act_sign) enables the symmetry loss.
  `train_policy` = False trains only the critic (warm-up after a warm start)."""
  obs_n = norm.normalize(obs_stats, mb["obs"], cfg.obs_clip)
  priv_n = norm.normalize(priv_stats, mb["priv"], cfg.obs_clip)
  mean, log_std = nets.policy_dist(params["policy"], obs_n, cfg.activation)
  logp = nets.log_prob(mean, log_std, mb["action"])
  ratio = jnp.exp(logp - mb["logp"])
  adv = mb["adv"]
  adv = (adv - adv.mean()) / (adv.std() + 1e-8)
  surr = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)
  policy_loss = -surr.mean()

  v = nets.value(params["value"], priv_n, cfg.activation)
  v_clip = mb["value"] + jnp.clip(v - mb["value"], -cfg.value_clip_eps, cfg.value_clip_eps)
  value_loss = jnp.maximum(jnp.square(v - mb["ret"]), jnp.square(v_clip - mb["ret"])).mean()

  ent = nets.entropy(log_std).mean()
  # Actions are clipped to [-1, 1] by the env, so a mean beyond a bound gets no
  # gradient back from the surrogate: pull it back explicitly.
  bound_loss = jnp.square(jax.nn.relu(jnp.abs(mean) - 1.0)).sum(-1).mean()
  pol = jnp.asarray(train_policy, jnp.float32)
  loss = (pol * (policy_loss - cfg.entropy_coef * ent + cfg.bound_coef * bound_loss)
          + cfg.value_coef * value_loss)
  aux_sym = {}
  if mirror is not None and cfg.symmetry_coef > 0:
    op, os_, ap, as_ = mirror
    obs_m = norm.normalize(obs_stats, mb["obs"][..., op] * os_, cfg.obs_clip)
    mean_m, _ = nets.policy_dist(params["policy"], obs_m, cfg.activation)
    sym_loss = jnp.square(mean_m - mean[..., ap] * as_).sum(-1).mean()
    loss = loss + pol * cfg.symmetry_coef * sym_loss
    aux_sym["loss/symmetry"] = sym_loss
  kl = nets.kl_divergence(mb["mean"], jnp.broadcast_to(old_log_std, mean.shape), mean, log_std).mean()
  aux = {"loss/policy": policy_loss, "loss/value": value_loss, "loss/entropy": ent,
         "loss/bound": bound_loss, "policy/saturated": (jnp.abs(mean) > 1.0).mean(),
         "train/kl": kl, "train/clip_frac": (jnp.abs(ratio - 1) > cfg.clip_eps).mean(), **aux_sym}
  return loss, aux


def adapt_lr(lr, kl, cfg: PPOConfig):
  if not cfg.adaptive_lr:
    return lr
  lo, hi = cfg.lr_bounds
  lr = jnp.where(kl > 2 * cfg.desired_kl, jnp.maximum(lr / 1.5, lo), lr)
  return jnp.where((kl < cfg.desired_kl / 2) & (kl > 0), jnp.minimum(lr * 1.5, hi), lr)


def gradient_step(carry, mb, obs_stats, priv_stats, old_log_std, cfg, mirror=None, train_policy=True):
  params, opt, lr = carry
  (_, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
      params, mb, obs_stats, priv_stats, old_log_std, cfg, mirror, train_policy)
  grads, gnorm = optim.clip_by_global_norm(grads, cfg.max_grad_norm)
  # rsl_rl: adapt on the KL measured before the step (frozen policy: KL is 0, keep lr)
  lr = jnp.where(jnp.asarray(train_policy), adapt_lr(lr, aux["train/kl"], cfg), lr)
  params, opt = optim.adam_update(params, grads, opt, lr)
  aux["train/grad_norm"] = gnorm
  aux["train/lr"] = lr
  return (params, opt, lr), aux


# ------------------------------------------------------------------ minibatches
def local_minibatches(key, data: dict, n_dev: int, n_mb: int, mesh=None) -> dict:
  """Shuffle and split samples into minibatches without crossing devices.

  `data` leaves have a leading sample axis S whose consecutive blocks of S/n_dev
  live on consecutive devices. Each device's block is permuted independently and
  cut into n_mb pieces; minibatch m takes piece m from every device.
  Returns leaves shaped (n_mb, S/n_mb, ...).
  """
  S = jax.tree.leaves(data)[0].shape[0]
  per = S // n_dev
  if per * n_dev != S or per % n_mb:
    raise ValueError(f"{S} samples cannot be split into {n_dev} devices x {n_mb} minibatches")
  b = per // n_mb
  perms = jax.vmap(lambda k: jax.random.permutation(k, per))(jax.random.split(key, n_dev))

  def constrain(x, spec):
    return lax.with_sharding_constraint(x, NamedSharding(mesh, spec)) if mesh is not None else x

  def split(x):
    rest = x.shape[1:]
    x = constrain(x.reshape(n_dev, per, *rest), P(AXIS))
    x = jax.vmap(lambda xi, pi: xi[pi])(x, perms)
    x = x.reshape(n_dev, n_mb, b, *rest).swapaxes(0, 1)  # (n_mb, n_dev, b, ...)
    x = constrain(x, P(None, AXIS))
    return x.reshape(n_mb, n_dev * b, *rest)

  return jax.tree.map(split, data)


# ------------------------------------------------------------------ train step
def make_train_step(env, cfg: PPOConfig, mesh):
  """Build the jitted PPO iteration: (TrainState, env State) -> (TrainState, env State, metrics)."""
  n_dev = mesh.devices.size
  T, N = cfg.unroll_length, cfg.num_envs
  if N % n_dev:
    raise ValueError(f"num_envs={N} must be divisible by the {n_dev} devices")
  act = cfg.activation
  w_track = env.cfg.reward.weights.get("tracking_lin_vel", 0.0) * env.dt
  w_yaw = env.cfg.reward.weights.get("tracking_ang_vel", 0.0) * env.dt
  batch_spec = NamedSharding(mesh, P(AXIS))
  mirror = None
  if cfg.symmetry_coef > 0:
    mirror = tuple(jnp.asarray(x) for x in symmetry.for_env(env))

  def train_step(ts: TrainState, env_state):
    key, k_roll, k_perm = jax.random.split(ts.key, 3)
    train_policy = ts.iteration >= cfg.critic_warmup_iters  # warm start: critic first
    env_state = env.set_dr_level(env_state, ts.dr_level)

    # ---- rollout ------------------------------------------------------------
    def env_step(carry, _):
      s, k = carry
      k, ka = jax.random.split(k)
      obs_n = norm.normalize(ts.obs_stats, s.obs, cfg.obs_clip)
      mean, log_std = nets.policy_dist(ts.params["policy"], obs_n, act)
      a = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
      logp = nets.log_prob(mean, log_std, a)
      s2 = jax.vmap(env.step)(s, a)
      tr = {"obs": s.obs, "priv": s.priv_obs, "action": a, "logp": logp, "mean": mean,
            "reward": s2.reward, "done": s2.done, "trunc": s2.truncation}
      return (s2, k), (tr, s2.metrics)

    (env_state, _), (traj, mets) = lax.scan(env_step, (env_state, k_roll), None, length=T)

    # ---- targets -------------------------------------------------------------
    # The loss uses the same normaliser statistics as the rollout, so at the first
    # gradient step the new and old policies are identical (ratio 1, KL 0). The
    # statistics are updated with this rollout's data only after optimisation.
    obs_stats, priv_stats = ts.obs_stats, ts.priv_stats
    all_priv = jnp.concatenate([traj["priv"], env_state.priv_obs[None]], axis=0)
    values = nets.value(ts.params["value"], norm.normalize(priv_stats, all_priv, cfg.obs_clip), act)
    adv, ret = compute_gae(traj["reward"], values, traj["done"], traj["trunc"],
                           cfg.gamma, cfg.gae_lambda)
    data = {k: traj[k] for k in ("obs", "priv", "action", "logp", "mean")}
    data.update(adv=adv, ret=ret, value=values[:-1])
    # (T, N, ...) -> (N*T, ...), env-major so each device's samples stay contiguous.
    data = jax.tree.map(lambda x: lax.with_sharding_constraint(
        x.swapaxes(0, 1).reshape(N * T, *x.shape[2:]), batch_spec), data)
    old_log_std = jnp.clip(ts.params["policy"]["log_std"], nets.LOG_STD_MIN, nets.LOG_STD_MAX)

    # ---- optimisation -----------------------------------------------------------
    def epoch(carry, k):
      mbs = local_minibatches(k, data, n_dev, cfg.num_minibatches, mesh)
      return lax.scan(lambda c, mb: gradient_step(c, mb, obs_stats, priv_stats, old_log_std, cfg, mirror,
                                                  train_policy),
                      carry, mbs)

    (params, opt, lr), aux = lax.scan(epoch, (ts.params, ts.opt, ts.lr),
                                      jax.random.split(k_perm, cfg.num_epochs))

    # ---- metrics (scalars only) -----------------------------------------------
    steps = float(N * T)
    msum = {k: v.sum() for k, v in mets.items()}
    n_ep = msum["episode/count"]
    # Means over all epochs x minibatches. The adaptive LR rises and falls within an
    # iteration (KL is measured against the rollout policy, so it grows as the
    # epochs proceed), so the mean LR is more informative than its final value.
    out = {k: v.mean() for k, v in aux.items()}
    out["train/kl_max"] = aux["train/kl"].max()
    out.update({
        # NaN (logged as null / "n/a") when no episode ended during this rollout
        "episode/return": jnp.where(n_ep > 0, msum["episode/return"] / jnp.maximum(n_ep, 1), jnp.nan),
        "episode/length": jnp.where(n_ep > 0, msum["episode/length"] / jnp.maximum(n_ep, 1), jnp.nan),
        "episode/fall_frac": jnp.where(n_ep > 0, msum["episode/fell"] / jnp.maximum(n_ep, 1), jnp.nan),
        "episode/count": n_ep,
        "rollout/reward": traj["reward"].mean(),
        "rollout/falls_per_1k_steps": msum["episode/fell"] / steps * 1000,
        "policy/std": jnp.exp(old_log_std).mean(),
        "value/explained_var": 1 - jnp.var(ret - values[:-1]) / (jnp.var(ret) + 1e-8),
        "curriculum/dr_level": ts.dr_level,
    })
    for k, v in msum.items():
      if k.startswith(("reward/", "debug/")) and k not in ("debug/dr_level",):
        out[k] = v / steps

    # ---- curriculum: raise randomisation once the robot walks and rarely falls --
    track = msum.get("reward/tracking_lin_vel", jnp.zeros(())) / steps / max(w_track, 1e-9)
    ok = ((track >= cfg.curriculum_tracking)
          & (msum["episode/fell"] / steps <= 1.0 / cfg.curriculum_survival_steps))
    dr_level = jnp.where(ok, jnp.minimum(ts.dr_level + cfg.dr_level_step, 1.0), ts.dr_level)
    out["curriculum/tracking"] = track
    out["curriculum/tracking_yaw"] = (msum.get("reward/tracking_ang_vel", jnp.zeros(())) / steps
                                      / max(w_yaw, 1e-9))

    obs_stats = norm.update(obs_stats, traj["obs"])
    priv_stats = norm.update(priv_stats, traj["priv"])
    ts = TrainState(params, opt, obs_stats, priv_stats, lr, dr_level, key, ts.iteration + 1)
    return ts, env_state, out

  rep = NamedSharding(mesh, P())
  return jax.jit(train_step, in_shardings=(rep, batch_spec), out_shardings=(rep, batch_spec, rep),
                 donate_argnums=(0, 1))
