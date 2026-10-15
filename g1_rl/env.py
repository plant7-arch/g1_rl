"""G1 velocity-command walking environment (pure JAX / MJX).

All methods operate on ONE environment; callers vmap them (see vec_env.py).
`step` auto-resets: when an episode ends, the returned state already holds a
fresh episode, while `reward`, `done`, `truncation` and `metrics` describe the
step that just finished.

Observations
------------
Policy ("obs", what the real robot can sense; noisy):
  gyro(3) gravity_in_body(3) command(3) joint_pos-home(29) joint_vel(29)
  last_action(29) gait_phase cos/sin(4)                               = 100
Critic ("priv_obs", simulator-only extras; noise-free):
  clean policy obs(100) base_lin_vel_body(3) base_height(1) foot_contact(2)
  foot_height(2) foot_air_time(2) actuator_force/limit(29)            = 139
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from g1_rl import math as gm
from g1_rl import rewards, robot, sim
from g1_rl.config import EnvConfig

FOOT_CAPSULES = (("left_foot1_collision", "left_foot2_collision", "left_foot3_collision"),
                 ("right_foot1_collision", "right_foot2_collision", "right_foot3_collision"))


def _layout(parts):
  """Name -> slice into a flat observation vector, plus total size under '_size'."""
  out, i = {}, 0
  for name, n in parts:
    out[name] = slice(i, i + n)
    i += n
  out["_size"] = i
  return out


class State(NamedTuple):
  data: tuple  # packed mjx.Data (see sim.DataPacker)
  obs: jax.Array
  priv_obs: jax.Array
  reward: jax.Array
  done: jax.Array  # 1.0 if the episode ended at this step (fall or timeout)
  truncation: jax.Array  # 1.0 if it ended because of the time limit
  info: dict
  metrics: dict


class G1Env:

  def __init__(self, cfg: EnvConfig = EnvConfig(), mj_model=None):
    self.cfg = cfg
    self.mj_model = mj_model if mj_model is not None else robot.load_mj_model(cfg.sim)
    m = self.mj_model
    self.robot = robot.robot_info(m, cfg.sim.home_key)
    self.mx = robot.make_mjx_model(m, cfg.sim)
    self.packer = sim.DataPacker(self.mx)
    self.n_substeps = cfg.sim.n_substeps
    self.dt = cfg.sim.ctrl_dt

    r = self.robot
    self.nu = r.nu
    self._home_q = np.asarray(r.home_qpos, np.float32)
    self._home_u = np.asarray(r.home_ctrl, np.float32)
    self._lo, self._hi = r.ctrl_lo, r.ctrl_hi
    self._qadr = r.actuator_qpos_adr
    self._vadr = r.actuator_qvel_adr
    self._torque_limit = np.maximum(m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1], 1.0).astype(np.float32)
    self._feet_site = r.feet_site_ids
    caps = np.array([[m.geom(n).id for n in foot] for foot in FOOT_CAPSULES], np.int32)
    self._foot_caps = caps  # (2, 3)
    self._cap_radius = m.geom_size[caps, 0].astype(np.float32)
    self._cap_halflen = m.geom_size[caps, 1].astype(np.float32)

    self.action_size = r.nu
    nu = r.nu
    self.obs_layout = _layout([("gyro", 3), ("gravity", 3), ("command", 3), ("joint_pos", nu),
                               ("joint_vel", nu), ("last_act", nu), ("gait", 4)])
    self.obs_size = self.obs_layout["_size"]
    self.priv_layout = _layout([("obs", self.obs_size), ("lin_vel", 3), ("height", 1),
                                ("contact", 2), ("foot_height", 2), ("air_time", 2),
                                ("act_force", nu)])
    self.priv_obs_size = self.priv_layout["_size"]

  # ------------------------------------------------------------------ helpers
  def foot_heights(self, d):
    """Lowest point of each foot's collision capsules above the floor, (2,)."""
    pos = d.geom_xpos[self._foot_caps]  # (2,3,3)
    axis_z = d.geom_xmat[self._foot_caps][..., 2, 2]  # z-component of capsule axis
    lowest = pos[..., 2] - jnp.abs(axis_z) * self._cap_halflen - self._cap_radius
    return lowest.min(axis=-1)

  def _sample_command(self, rng):
    c = self.cfg.command
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    lo = jnp.array([c.lin_vel_x[0], c.lin_vel_y[0], c.ang_vel_z[0]])
    hi = jnp.array([c.lin_vel_x[1], c.lin_vel_y[1], c.ang_vel_z[1]])
    cmd = jax.random.uniform(k1, (3,), minval=lo, maxval=hi)
    cmd = jnp.where(jnp.abs(cmd) < c.deadband, 0.0, cmd)
    cmd = jnp.where(jax.random.uniform(k2) < c.zero_prob, jnp.zeros(3), cmd)
    timer = jax.random.randint(k3, (), c.resample_steps[0], c.resample_steps[1] + 1)
    return cmd, timer

  def _observe(self, d, info, rng):
    q, v = d.qpos, d.qvel
    quat = q[3:7]
    gyro = v[3:6]  # free-joint angular velocity is already in the body frame
    gravity = gm.quat_rotate_inverse(quat, jnp.array([0.0, 0.0, -1.0]))
    jpos = q[self._qadr] - self._home_u
    jvel = v[self._vadr]
    phase = info["phase"]
    gait = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)])
    clean = jnp.concatenate([gyro, gravity, info["command"], jpos, jvel, info["last_act"], gait])

    n = self.cfg.noise
    scale = jnp.concatenate([
        jnp.full(3, n.gyro), jnp.full(3, n.gravity), jnp.zeros(3),
        jnp.full(self.nu, n.joint_pos), jnp.full(self.nu, n.joint_vel),
        jnp.zeros(self.nu), jnp.zeros(4)]) * n.level
    noisy = clean + scale * jax.random.uniform(rng, clean.shape, minval=-1.0, maxval=1.0)

    lin_vel = gm.quat_rotate_inverse(quat, v[0:3])
    fh = self.foot_heights(d)
    priv = jnp.concatenate([
        clean, lin_vel, q[2:3], info["contact"].astype(jnp.float32), fh,
        info["feet_air_time"], d.actuator_force / self._torque_limit])
    return jnp.nan_to_num(noisy), jnp.nan_to_num(priv)

  def _fresh_episode(self, d, rng):
    """New initial configuration written into an existing Data (no full forward)."""
    cfg = self.cfg
    ks = jax.random.split(rng, 8)
    nu = self.nu
    q = jnp.asarray(self._home_q)
    xy = jax.random.uniform(ks[0], (2,), minval=-cfg.init_xy_range, maxval=cfg.init_xy_range)
    yaw = jax.random.uniform(ks[1], (), minval=-jnp.pi, maxval=jnp.pi)
    q = q.at[0:2].add(xy)
    q = q.at[3:7].set(gm.quat_mul(gm.yaw_quat(yaw), q[3:7]))
    q = q.at[self._qadr].add(jax.random.uniform(
        ks[2], (nu,), minval=-cfg.init_joint_noise, maxval=cfg.init_joint_noise))
    q = q.at[self._qadr].set(jnp.clip(q[self._qadr], self._lo, self._hi))
    v = jnp.zeros(self.mx.nv)
    v = v.at[0:6].set(jax.random.uniform(
        ks[3], (6,), minval=-cfg.init_base_vel_noise, maxval=cfg.init_base_vel_noise))

    d = d.replace(qpos=q, qvel=v, ctrl=jnp.asarray(self._home_u), time=jnp.zeros(()),
                  qacc_warmstart=jnp.zeros_like(d.qacc_warmstart),
                  qfrc_applied=jnp.zeros_like(d.qfrc_applied),
                  xfrc_applied=jnp.zeros_like(d.xfrc_applied),
                  actuator_force=jnp.zeros_like(d.actuator_force))
    # Kinematics only (cheap): positions for feet/obs. Dynamics are recomputed by
    # mjx.step. Shift the base so the lowest foot point rests exactly on the floor
    # (joint noise would otherwise spawn the robot floating or penetrating).
    d = mjx.kinematics(self.mx, d)
    q = q.at[2].add(-self.foot_heights(d).min())
    d = mjx.kinematics(self.mx, d.replace(qpos=q))

    cmd, timer = self._sample_command(ks[4])
    freq = jax.random.uniform(ks[5], (), minval=cfg.gait_freq[0], maxval=cfg.gait_freq[1])
    info = {
        "rng": ks[6],
        "step": jnp.zeros((), jnp.int32),
        "command": cmd,
        "cmd_timer": timer,
        "phase": jnp.array([0.0, jnp.pi]),
        "phase_dt": 2 * jnp.pi * freq * self.dt,
        "last_act": jnp.zeros(nu),
        "last_last_act": jnp.zeros(nu),
        "motor_targets": jnp.asarray(self._home_u),
        "contact": jnp.ones(2, bool),
        "feet_air_time": jnp.zeros(2),
        "swing_peak": jnp.zeros(2),
        "ep_return": jnp.zeros(()),
    }
    return d, info, ks[7]

  def _reward_terms(self, d, d_prev, info, prev_info, action, contact, first_contact, fh, fell):
    ctx = dict(env=self, d=d, d_prev=d_prev, info=info, prev_info=prev_info, action=action,
               contact=contact, first_contact=first_contact, foot_heights=fh, fell=fell)
    terms = rewards.compute(ctx, self.cfg)
    return {k: jnp.where(jnp.isfinite(v), v, 0.0) for k, v in terms.items()}

  @staticmethod
  def _metrics(terms, ep_return, ep_len, fell, done):
    m = {f"reward/{k}": v for k, v in terms.items()}
    m.update({"episode/return": ep_return * done, "episode/length": ep_len * done,
              "episode/fell": fell * done, "episode/count": done})
    return m

  # ---------------------------------------------------------------- public API
  def reset(self, rng) -> State:
    d = sim.init_data(self.mx, jnp.asarray(self._home_q), jnp.asarray(self._home_u))
    d, info, obs_rng = self._fresh_episode(d, rng)
    obs, priv = self._observe(d, info, obs_rng)
    zero = jnp.zeros(())
    terms = self._reward_terms(d, d, info, info, jnp.zeros(self.nu), info["contact"],
                               info["contact"] & False, self.foot_heights(d), jnp.array(False))
    metrics = self._metrics({k: zero for k in terms}, zero, zero, zero, zero)
    return State(self.packer.pack(d), obs, priv, zero, zero, zero, info, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    cfg = self.cfg
    info = dict(state.info)
    rng, k_obs, k_cmd, k_reset = jax.random.split(info["rng"], 4)
    d_prev = self.packer.unpack(state.data)

    action = jnp.clip(action, -1.0, 1.0)
    targets = jnp.clip(self._home_u + cfg.action_scale * action, self._lo, self._hi)
    d = sim.step_ctrl(self.mx, d_prev, targets, self.n_substeps)

    # --- bookkeeping -------------------------------------------------------
    fh = self.foot_heights(d)
    contact = fh < cfg.contact_threshold
    first_contact = (info["feet_air_time"] > 0) & contact
    info["feet_air_time"] = (info["feet_air_time"] + self.dt) * ~contact
    info["swing_peak"] = jnp.where(contact, 0.0, jnp.maximum(info["swing_peak"], fh))
    info["phase"] = gm.wrap_to_pi(info["phase"] + info["phase_dt"])
    info["step"] = info["step"] + 1
    info["cmd_timer"] = info["cmd_timer"] - 1
    new_cmd, new_timer = self._sample_command(k_cmd)
    resample = info["cmd_timer"] <= 0
    info["command"] = jnp.where(resample, new_cmd, info["command"])
    info["cmd_timer"] = jnp.where(resample, new_timer, info["cmd_timer"])

    # --- termination -------------------------------------------------------
    up_z = gm.quat_rotate(d.qpos[3:7], jnp.array([0.0, 0.0, 1.0]))[2]
    fell = (d.qpos[2] < cfg.min_base_height) | (up_z < cfg.max_tilt_cos)
    fell = fell | ~jnp.all(jnp.isfinite(d.qpos)) | ~jnp.all(jnp.isfinite(d.qvel))
    timeout = info["step"] >= cfg.episode_length
    done = fell | timeout

    # --- reward --------------------------------------------------------------
    terms = self._reward_terms(d, d_prev, info, state.info, action, contact, first_contact,
                               fh, fell)
    reward = rewards.total(terms, self.dt)

    info["last_last_act"] = info["last_act"]
    info["last_act"] = action
    info["motor_targets"] = targets
    info["contact"] = contact
    info["ep_return"] = info["ep_return"] + reward
    info["rng"] = rng

    obs, priv = self._observe(d, info, k_obs)

    f_done = done.astype(jnp.float32)
    metrics = self._metrics(terms, info["ep_return"], info["step"].astype(jnp.float32),
                            fell.astype(jnp.float32), f_done)

    # --- auto-reset ------------------------------------------------------------
    d_new, info_new, k_obs2 = self._fresh_episode(d, k_reset)
    obs_new, priv_new = self._observe(d_new, info_new, k_obs2)
    d = sim.tree_where(done, d_new, d)
    info = sim.tree_where(done, info_new, info)
    obs = jnp.where(done, obs_new, obs)
    priv = jnp.where(done, priv_new, priv)

    return State(self.packer.pack(d), obs, priv, reward, f_done,
                 (timeout & ~fell).astype(jnp.float32), info, metrics)
