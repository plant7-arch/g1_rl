"""Phase 4: PPO building blocks on synthetic data (no simulator, ~1 min on CPU)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g1_rl import networks as nets
from g1_rl import normalizer as norm
from g1_rl import optim, ppo
from g1_rl.config import PPOConfig

CFG = PPOConfig()


def test_gae_matches_reference_loop():
  rng = np.random.default_rng(0)
  T, N, g, lam = 7, 3, 0.9, 0.8
  r = rng.normal(size=(T, N)).astype(np.float32)
  v = rng.normal(size=(T + 1, N)).astype(np.float32)
  done = np.zeros((T, N), np.float32)
  trunc = np.zeros((T, N), np.float32)
  done[2, 0] = 1  # fall
  done[4, 1] = trunc[4, 1] = 1  # time limit
  adv_ref = np.zeros((T, N), np.float32)
  last = np.zeros(N, np.float32)
  for t in reversed(range(T)):
    rt = r[t] + g * v[t] * trunc[t]
    nd = 1 - done[t]
    delta = rt + g * v[t + 1] * nd - v[t]
    last = delta + g * lam * nd * last
    adv_ref[t] = last
  adv, ret = ppo.compute_gae(jnp.asarray(r), jnp.asarray(v), jnp.asarray(done), jnp.asarray(trunc), g, lam)
  np.testing.assert_allclose(np.asarray(adv), adv_ref, rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(np.asarray(ret), adv_ref + v[:-1], rtol=1e-5, atol=1e-6)
  # A fall cuts the bootstrap: the advantage at the fall step only sees its own reward.
  assert np.isclose(adv_ref[2, 0], r[2, 0] - v[2, 0])


def test_gaussian_formulas():
  rng = np.random.default_rng(1)
  m, ls, x = (rng.normal(size=(5, 4)).astype(np.float32) for _ in range(3))
  ls = 0.3 * ls
  s = np.exp(ls)
  ref = np.sum(-0.5 * ((x - m) / s) ** 2 - ls - 0.5 * np.log(2 * np.pi), -1)
  np.testing.assert_allclose(np.asarray(nets.log_prob(m, ls, x)), ref, rtol=1e-5)
  np.testing.assert_allclose(np.asarray(nets.entropy(ls)),
                             np.sum(0.5 * np.log(2 * np.pi * np.e * s ** 2), -1), rtol=1e-5)
  m2, ls2 = m + 0.1, ls - 0.2
  s2 = np.exp(ls2)
  kl_ref = np.sum(np.log(s2 / s) + (s ** 2 + (m - m2) ** 2) / (2 * s2 ** 2) - 0.5, -1)
  np.testing.assert_allclose(np.asarray(nets.kl_divergence(m, ls, m2, ls2)), kl_ref, rtol=1e-4)
  assert np.allclose(np.asarray(nets.kl_divergence(m, ls, m, ls)), 0, atol=1e-6)


def test_adam_matches_reference_and_clipping():
  p = {"a": jnp.array([1.0, -2.0]), "b": jnp.array(3.0)}
  st = optim.adam_init(p)
  m = {k: np.zeros_like(np.asarray(v)) for k, v in p.items()}
  vv = {k: np.zeros_like(np.asarray(v)) for k, v in p.items()}
  ref = {k: np.asarray(v).copy() for k, v in p.items()}
  for t in range(1, 6):
    g = {"a": jnp.array([0.5, -1.0]) * t, "b": jnp.array(0.2 * t)}
    p, st = optim.adam_update(p, g, st, 0.01)
    for k in ref:
      gk = np.asarray(g[k])
      m[k] = 0.9 * m[k] + 0.1 * gk
      vv[k] = 0.999 * vv[k] + 0.001 * gk ** 2
      ref[k] = ref[k] - 0.01 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(vv[k] / (1 - 0.999 ** t)) + 1e-8)
  for k in ref:
    np.testing.assert_allclose(np.asarray(p[k]), ref[k], rtol=1e-5)
  g, n = optim.clip_by_global_norm({"x": jnp.array([3.0, 4.0])}, 1.0)
  assert float(n) == pytest.approx(5.0) and float(optim.global_norm(g)) == pytest.approx(1.0, rel=1e-4)


def test_normalizer_merges_batches_exactly():
  rng = np.random.default_rng(2)
  chunks = [rng.normal(loc=3, scale=2, size=(n, 6)).astype(np.float32) for n in (50, 7, 300)]
  st = norm.init(6)
  for c in chunks:
    st = norm.update(st, jnp.asarray(c))
  allx = np.concatenate(chunks)
  np.testing.assert_allclose(np.asarray(st.mean), allx.mean(0), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(np.asarray(st.var), allx.var(0), rtol=1e-3)
  z = np.asarray(norm.normalize(st, jnp.asarray(allx)))
  assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01


def test_local_minibatches_use_every_sample_once_and_stay_on_device():
  n_dev, n_mb, S = 2, 4, 48
  data = {"id": jnp.arange(S), "x": jnp.arange(S * 3).reshape(S, 3)}
  mbs = ppo.local_minibatches(jax.random.PRNGKey(0), data, n_dev, n_mb)
  ids = np.asarray(mbs["id"])
  assert ids.shape == (n_mb, S // n_mb)
  assert sorted(ids.ravel().tolist()) == list(range(S))  # each sample exactly once
  half = S // n_dev
  for m in range(n_mb):  # first half of each minibatch from device 0's block, rest from device 1's
    row = ids[m]
    assert (row[: len(row) // 2] < half).all() and (row[len(row) // 2:] >= half).all()
  np.testing.assert_array_equal(np.asarray(mbs["x"])[..., 0], ids * 3)  # leaves permuted together
  with pytest.raises(ValueError):
    ppo.local_minibatches(jax.random.PRNGKey(0), {"id": jnp.arange(50)}, 2, 4)


def _toy_batch(key, params, stats, cfg, B=512, dim=4, nu=3):
  """Contextual bandit: reward = -|a - W obs|^2 (one step, so return = reward)."""
  W = jnp.arange(dim * nu, dtype=jnp.float32).reshape(dim, nu) / (dim * nu)
  ko, ka = jax.random.split(key)
  obs = jax.random.normal(ko, (B, dim))
  mean, log_std = nets.policy_dist(params["policy"], norm.normalize(stats, obs), cfg.activation)
  a = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
  r = -jnp.sum(jnp.square(a - obs @ W), -1)
  v = nets.value(params["value"], norm.normalize(stats, obs), cfg.activation)
  return {"obs": obs, "priv": obs, "action": a, "logp": nets.log_prob(mean, log_std, a),
          "mean": mean, "adv": r - v, "ret": r, "value": v}, r.mean()


def test_ppo_loss_basics():
  cfg = dataclasses.replace(CFG, policy_hidden=(16,), value_hidden=(16,))
  params = nets.init_actor_critic(jax.random.PRNGKey(0), 4, 4, 3, (16,), (16,), 0.6)
  stats = norm.init(4)
  mb, _ = _toy_batch(jax.random.PRNGKey(1), params, stats, cfg)
  old_ls = params["policy"]["log_std"]
  _, aux = ppo.ppo_loss(params, mb, stats, stats, old_ls, cfg)
  assert float(aux["train/kl"]) == pytest.approx(0, abs=1e-6)  # same policy
  assert float(aux["train/clip_frac"]) == 0
  assert np.isfinite([float(v) for v in aux.values()]).all()


def test_ppo_learns_a_toy_bandit():
  cfg = dataclasses.replace(CFG, policy_hidden=(32, 32), value_hidden=(32, 32), init_std=0.5)
  params = nets.init_actor_critic(jax.random.PRNGKey(0), 4, 4, 3, (32, 32), (32, 32), 0.5)
  opt = optim.adam_init(params)
  stats = norm.init(4)
  lr = jnp.asarray(1e-3)
  step = jax.jit(lambda c, mb: ppo.gradient_step(c, mb, stats, stats, c[0]["policy"]["log_std"], cfg))
  batch = jax.jit(lambda k, p: _toy_batch(k, p, stats, cfg))
  key = jax.random.PRNGKey(5)
  rewards = []
  for it in range(60):
    key, kb, kp = jax.random.split(key, 3)
    data, r = batch(kb, params)
    rewards.append(float(r))
    old_ls = params["policy"]["log_std"]
    for _ in range(4):  # epochs
      mbs = ppo.local_minibatches(kp, data, 1, 4)
      for m in range(4):
        (params, opt, lr), aux = step((params, opt, lr), jax.tree.map(lambda x: x[m], mbs))
  assert rewards[-1] > 0.25 * rewards[0]  # error shrinks by > 4x (rewards are negative)
  assert cfg.lr_bounds[0] <= float(lr) <= cfg.lr_bounds[1]


def test_adaptive_lr():
  lr = jnp.asarray(1e-3)
  assert float(ppo.adapt_lr(lr, jnp.asarray(0.1), CFG)) == pytest.approx(1e-3 / 1.5)
  assert float(ppo.adapt_lr(lr, jnp.asarray(0.001), CFG)) == pytest.approx(1.5e-3)
  assert float(ppo.adapt_lr(lr, jnp.asarray(0.01), CFG)) == pytest.approx(1e-3)
  assert float(ppo.adapt_lr(jnp.asarray(1e-5), jnp.asarray(1.0), CFG)) == pytest.approx(1e-5)


def test_bound_and_symmetry_losses():
  """Bound loss = sum over action dims of (|mean| - 1)^2 beyond the bounds; symmetry
  loss = |mu(mirror(obs)) - mirror(mu(obs))|^2, zero for a policy that is equivariant."""
  cfg = dataclasses.replace(CFG, policy_hidden=(16,), value_hidden=(16,))
  params = nets.init_actor_critic(jax.random.PRNGKey(0), 4, 4, 3, (16,), (16,), 0.6)
  stats = norm.init(4)
  mb, _ = _toy_batch(jax.random.PRNGKey(1), params, stats, cfg)
  old_ls = params["policy"]["log_std"]
  net = [dict(l) for l in params["policy"]["net"]]
  net[-1]["w"] = jnp.zeros_like(net[-1]["w"])
  net[-1]["b"] = jnp.array([2.0, -1.5, 0.5])  # mean = (2, -1.5, 0.5) everywhere
  p2 = {**params, "policy": {**params["policy"], "net": net}}
  _, aux = ppo.ppo_loss(p2, mb, stats, stats, old_ls, cfg)
  assert float(aux["loss/bound"]) == pytest.approx(1.0 + 0.25, rel=1e-5)
  assert float(aux["policy/saturated"]) == pytest.approx(2 / 3, rel=1e-5)
  # Gradient of the bound loss pulls the out-of-range means back inside.
  g = jax.grad(lambda p: ppo.ppo_loss(p, mb, stats, stats, old_ls,
                                     dataclasses.replace(cfg, value_coef=0, entropy_coef=0))[0])(p2)
  gb = g["policy"]["net"][-1]["b"]
  assert float(gb[0]) > 0 and float(gb[1]) < 0  # descent lowers mean 0, raises mean 1

  ident = (jnp.arange(4), jnp.ones(4), jnp.arange(3), jnp.ones(3))
  _, aux = ppo.ppo_loss(params, mb, stats, stats, old_ls, cfg, mirror=ident)
  assert float(aux["loss/symmetry"]) == pytest.approx(0.0, abs=1e-10)
  swap = (jnp.array([1, 0, 2, 3]), jnp.array([1.0, 1.0, -1.0, 1.0]), jnp.array([1, 0, 2]),
          jnp.array([1.0, 1.0, -1.0]))
  _, aux = ppo.ppo_loss(params, mb, stats, stats, old_ls, cfg, mirror=swap)
  obs_n = norm.normalize(stats, mb["obs"], cfg.obs_clip)
  mu = nets.policy_dist(params["policy"], obs_n, cfg.activation)[0]
  mu_m = nets.policy_dist(params["policy"], norm.normalize(stats, mb["obs"][:, swap[0]] * swap[1],
                                                           cfg.obs_clip), cfg.activation)[0]
  want = jnp.square(mu_m - mu[:, swap[2]] * swap[3]).sum(-1).mean()
  assert float(aux["loss/symmetry"]) == pytest.approx(float(want), rel=1e-5) and float(want) > 0
