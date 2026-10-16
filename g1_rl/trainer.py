"""Training loop around ppo.make_train_step: resume, logging, checkpoints, time budget, video.

The host only (1) launches the jitted iteration, (2) fetches the previous
iteration's ~50 scalar metrics while the next one runs on the GPUs, and
(3) every `ckpt_every` iterations copies the parameters to disk.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import g1_rl
from g1_rl import checkpoint, parallel, ppo
from g1_rl.config import EnvConfig, PPOConfig
from g1_rl.env import G1Env
from g1_rl.logger import Logger
from g1_rl.vec_env import VecEnv


def _validate(cfg: PPOConfig, n_dev: int):
  if cfg.num_envs % n_dev:
    raise ValueError(f"num_envs={cfg.num_envs} not divisible by {n_dev} devices")
  per = cfg.num_envs // n_dev * cfg.unroll_length
  if per % cfg.num_minibatches:
    raise ValueError(f"per-device samples {per} not divisible by num_minibatches={cfg.num_minibatches}")


def _asdict(cfg):
  return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


def train(ppo_cfg: PPOConfig, run_dir: str, rt, env_cfg: EnvConfig = EnvConfig(),
          resume: str | None = None, max_hours: float | None = None, video: bool = True,
          stop_after: int | None = None) -> dict:
  """Train (or resume) a policy. Returns a summary dict.

  resume: run dir or checkpoint to continue from (may live in another directory,
          e.g. a previous Kaggle version's output); new files go to `run_dir`.
  max_hours: stop cleanly (with a checkpoint) before this much wall-clock time.
  stop_after: stop after this many iterations in this call (tests / smoke runs).
  """
  t_start = time.time()
  _validate(ppo_cfg, rt.n_devices)
  os.makedirs(run_dir, exist_ok=True)
  env = G1Env(env_cfg)
  template = ppo.init_train_state(jax.random.PRNGKey(ppo_cfg.seed), env.obs_size,
                                  env.priv_obs_size, env.action_size, ppo_cfg)
  ts, start_it = template, 0
  if resume:
    ts, meta, path = checkpoint.load_checkpoint(resume, template)
    start_it = int(ts.iteration)
    print(f"resumed from {path} at iteration {start_it} "
          f"(dr_level {float(ts.dr_level):.2f}, lr {float(ts.lr):.1e})")
  n_iters = ppo_cfg.num_iterations
  if start_it >= n_iters:
    print(f"nothing to do: checkpoint is at iteration {start_it} of {n_iters}")
  ts = parallel.put_replicated(ts, rt.mesh)

  venv = VecEnv(env, rt.mesh, ppo_cfg.num_envs)
  env_state = venv.reset(seed=ppo_cfg.seed * 100_003 + start_it, dr_level=float(ts.dr_level))
  step = ppo.make_train_step(env, ppo_cfg, rt.mesh)

  meta = {"g1_rl_version": g1_rl.__version__, "jax": jax.__version__, "runtime": rt.describe(),
          "ppo": _asdict(ppo_cfg), "env": _asdict(env_cfg)}
  with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(dict(meta, resumed_from=resume, num_iterations=n_iters), f, indent=2)
  print(f"run dir: {run_dir}\n{rt.describe()} | {ppo_cfg.num_envs} envs x {ppo_cfg.unroll_length} "
        f"steps = {ppo_cfg.batch_size:,} per iteration | {n_iters} iterations "
        f"({ppo_cfg.total_env_steps / 1e6:.0f}M steps) | obs {env.obs_size} / critic {env.priv_obs_size}")

  log = Logger(run_dir, n_iters, ppo_cfg.batch_size, ppo_cfg.log_every, start_it)
  deadline = t_start + max_hours * 3600 if max_hours else math.inf
  pending, last_metrics, stopped, iter_time = None, {}, "finished", 0.0

  def flush():
    nonlocal pending, last_metrics, stopped
    if pending is None:
      return
    it, m = pending
    pending = None
    m = {k: float(v) for k, v in jax.device_get(m).items()}
    last_metrics = m
    log.log(it, m)
    if not all(math.isfinite(m[k]) for k in ("loss/policy", "loss/value")):
      stopped = "diverged"

  def save(tag: str):
    ckpt = checkpoint.save_checkpoint(run_dir, ts, dict(meta, env_steps=int(ts.iteration) * ppo_cfg.batch_size),
                                      keep=ppo_cfg.keep_ckpts)
    checkpoint.export_policy(os.path.join(run_dir, f"policy_{tag}.npz"), ts, env, ppo_cfg,
                             {"env_steps": int(ts.iteration) * ppo_cfg.batch_size})
    return ckpt

  it = start_it
  for it in range(start_it, n_iters):
    t_it = time.time()
    if it == start_it:
      print("compiling the training step (first iteration takes a few minutes)...", flush=True)
    ts, env_state, m = step(ts, env_state)
    flush()  # previous iteration's metrics, fetched while this one runs
    pending = (it + 1, m)
    if it == start_it:
      jax.block_until_ready(m)
      print(f"first iteration (incl. compile) took {time.time() - t_it:.0f}s", flush=True)
    if stopped == "diverged":
      print(f"!! non-finite loss at iteration {it}; stopping. Last saved checkpoint is intact.")
      break
    if (it + 1) % ppo_cfg.ckpt_every == 0:
      flush()
      save("latest")
    iter_time = 0.9 * iter_time + 0.1 * (time.time() - t_it) if iter_time else time.time() - t_it
    if time.time() + 2 * iter_time + 120 > deadline:
      stopped = "time budget"
      break
    if stop_after is not None and it + 1 - start_it >= stop_after:
      stopped = "stop_after"
      break
  flush()
  log.close()

  done_it = int(jax.device_get(ts.iteration))
  if stopped != "diverged" and done_it > start_it:
    ckpt = save("latest")
    if done_it >= n_iters:
      checkpoint.export_policy(os.path.join(run_dir, "policy_final.npz"), ts, env, ppo_cfg,
                               {"env_steps": done_it * ppo_cfg.batch_size})
    print(f"saved {ckpt}")
  summary = {"stopped": stopped, "iterations_done": done_it, "of": n_iters,
             "env_steps": done_it * ppo_cfg.batch_size, "hours": (time.time() - t_start) / 3600,
             "final_metrics": last_metrics}
  with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
  print(f"stopped: {stopped} after {done_it}/{n_iters} iterations "
        f"({summary['hours']:.2f} h this session)")

  if video and done_it > start_it and stopped != "diverged":
    try:
      record_video(env, ts, ppo_cfg, rt.mesh, os.path.join(run_dir, "video_walk_forward"))
    except Exception as e:  # never lose a finished run to a rendering problem
      print(f"[video] skipped: {type(e).__name__}: {str(e)[:200]}")
  return summary


def record_video(env, ts, ppo_cfg, mesh, out_base: str, seconds: float = 10.0,
                 command=(0.5, 0.0, 0.0)):
  """10 s of the deterministic policy on the nominal robot walking forward -> .npz + .mp4."""
  from g1_rl import networks as nets  # noqa: PLC0415
  from g1_rl import normalizer as norm  # noqa: PLC0415
  from g1_rl import render  # noqa: PLC0415

  n = mesh.devices.size
  venv = VecEnv(env, mesh, n)
  s = venv.reset(seed=123, dr_level=0.0)
  info = dict(s.info)
  info["command"] = jnp.tile(jnp.asarray(command, jnp.float32), (n, 1))
  info["cmd_timer"] = jnp.full_like(info["cmd_timer"], 10 ** 6)
  info["push_timer"] = jnp.full_like(info["push_timer"], 10 ** 6)
  s = parallel.put_batched(s._replace(info=info), mesh)
  params, stats = jax.device_get(ts.params["policy"]), jax.device_get(ts.obs_stats)

  def policy_fn(obs, key):
    mean, _ = nets.policy_dist(params, norm.normalize(stats, obs, ppo_cfg.obs_clip), ppo_cfg.activation)
    return mean

  steps = int(seconds / env.dt)
  roll = venv.make_rollout(policy_fn, steps, record=lambda st: env.packer.unpack(st.data).qpos)
  _, _, qpos, met = roll(s, venv.keys(7))
  qpos = np.asarray(jax.device_get(qpos))[0]
  falls = float(np.asarray(jax.device_get(met["episode/fell"]))[0])
  traj = render.save_trajectory(out_base + ".npz", qpos, fps=1 / env.dt, xml_path=env.cfg.sim.xml_path,
                                command=np.asarray(command), falls=falls)
  vid = render.try_render(traj, out_base + ".mp4")
  dist = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
  print(f"[video] {seconds:.0f}s forward-walk test: {falls:.0f} falls, moved {dist:.2f} m "
        f"(commanded {command[0] * seconds:.1f} m) -> {vid or traj}")
