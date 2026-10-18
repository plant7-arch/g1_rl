"""G1 velocity-command walking environment (pure JAX / MJX).

All methods operate on ONE environment; callers vmap them (see vec_env.py).
Rewards are computed by g1_rl/rewards.py from a RewardInputs built in
`_reward_terms`. `step` auto-resets: when an episode ends, the returned state already holds a
fresh episode, while `reward`, `done`, `truncation` and `metrics` describe the
step that just finished.

Observations
------------
Policy ("obs", what the real robot can sense; noisy). One frame:
  gyro(3) gravity_in_body(3) command(3) joint_pos-home(29) joint_vel(29)
  last_action(29) gait_phase cos/sin(4)                               = 100
and the policy sees the last `obs_history` frames, newest first (v0.9: 5 x 100 = 500).
`obs_layout` describes one frame; the current frame is obs[:frame_size].
The policy frame also carries this episode's sensor errors (IMU mounting tilt,
gyro bias, joint-encoder offsets) on top of the noise.
Critic ("priv_obs", simulator-only extras; noise-free):
  clean current frame(100) base_lin_vel_body(3) base_height(1) foot_contact(2)
  foot_height(2) foot_air_time(2) actuator_force/limit(29)
  randomised robot: friction(1) payload(1) com_offset(3) kp_scale(29)
  kd_scale(29) latency(1) recent_push(2) slope(2) imu_tilt(2) gyro_bias(3) = 212

Slopes (v0.9) are simulated by tilting gravity (info["dr"]["gravity_dir"]): the floor
stays the plane z = 0, so contacts, foot heights and velocities are all measured in
the slope's own frame, while "gravity in the body frame" uses the tilted direction.

Domain randomisation (randomize.py) is sampled per episode and stored in
info["dr"]; info["dr_level"] in [0, 1] scales it (curriculum, see set_dr_level).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from g1_rl import math as gm
from g1_rl import randomize, rewards, robot, sim
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


def _rot_roll_pitch(roll, pitch):
  cr, sr, cp, sp = jnp.cos(roll), jnp.sin(roll), jnp.cos(pitch), jnp.sin(pitch)
  rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
  ry = jnp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
  return rx @ ry


def stack_frames(frame, prev_obs=None, history: int = 1):
  """Policy observation = the last `history` frames, newest first. With no previous
  observation (a new episode) the current frame fills every slot."""
  if prev_obs is None:
    return jnp.tile(frame, history)
  return jnp.concatenate([frame, prev_obs[:-frame.shape[-1]]]) if history > 1 else frame


def action_scales(joint_names, cfg: EnvConfig) -> np.ndarray:
  """Per-joint action scale: cfg.action_scale unless a cfg.action_scale_overrides key
  is a substring of the joint name."""
  out = np.full(len(joint_names), cfg.action_scale, np.float32)
  for i, n in enumerate(joint_names):
    for k, v in cfg.action_scale_overrides.items():
      if k in n:
        out[i] = v
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
    self._rconsts = rewards.make_consts(r.actuator_names, self._home_u, self._lo, self._hi,
                                        cfg.reward, self.dt)
    self.dr = randomize.Randomizer(m, cfg.dr, r.torso_body_id, self.dt)
    caps = np.array([[m.geom(n).id for n in foot] for foot in FOOT_CAPSULES], np.int32)
    self._foot_caps = caps  # (2, 3)
    self._geom_body = np.asarray(m.geom_bodyid, np.int32)
    # Self-collision pairs that do not count for the penalty: hands against hips/thighs.
    # The training model's hand capsules are much larger than the real hand (the full mesh
    # model shows almost no such contact), and penalising them made g1_walk_v4 hold both
    # arms spread at the limit of their range (v1.0).
    names = [m.body(i).name for i in range(m.nbody)]
    hand = np.array([("wrist" in n) or ("hand" in n) or ("rubber" in n) for n in names])
    leg = np.array([("hip" in n) or ("knee" in n) or ("thigh" in n) for n in names])
    self._ignore_pair = (hand[:, None] & leg[None, :]) | (leg[:, None] & hand[None, :])
    self._cap_radius = m.geom_size[caps, 0].astype(np.float32)
    self._cap_halflen = m.geom_size[caps, 1].astype(np.float32)

    self.action_size = r.nu
    self.action_scale = action_scales(r.actuator_names, cfg)  # (nu,) rad per unit action
    nu = r.nu
    self.obs_layout = _layout([("gyro", 3), ("gravity", 3), ("command", 3), ("joint_pos", nu),
                               ("joint_vel", nu), ("last_act", nu), ("gait", 4)])
    self.frame_size = self.obs_layout["_size"]
    self.history = max(1, int(cfg.obs_history))
    self.obs_size = self.frame_size * self.history
    self.priv_layout = _layout([("obs", self.frame_size), ("lin_vel", 3), ("height", 1),
                                ("contact", 2), ("foot_height", 2), ("air_time", 2),
                                ("act_force", nu), ("dr", self.dr.priv_size)])
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
    # Some commands are drawn from a small range so slow walking/turning is practised.
    small = jnp.clip(jax.random.uniform(k4, (3,), minval=-1.0, maxval=1.0) * jnp.asarray(c.small_range),
                     lo, hi)
    cmd = jnp.where(jax.random.uniform(jax.random.fold_in(k4, 1)) < c.small_prob, small, cmd)
    cmd = jnp.where(jnp.abs(cmd) < c.deadband, 0.0, cmd)
    cmd = jnp.where(jax.random.uniform(k2) < c.zero_prob, jnp.zeros(3), cmd)
    timer = jax.random.randint(k3, (), c.resample_steps[0], c.resample_steps[1] + 1)
    return cmd, timer

  def _observe(self, d, info, rng):
    """(noisy policy frame, privileged critic obs) for the current state."""
    q, v = d.qpos, d.qvel
    quat = q[3:7]
    dr = info["dr"]
    gyro = v[3:6]  # free-joint angular velocity is already in the body frame
    gravity = gm.quat_rotate_inverse(quat, dr["gravity_dir"])
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
    # This episode's sensor errors (policy only): IMU mounted slightly tilted, gyro bias,
    # joint-encoder offsets.
    L = self.obs_layout
    rot = _rot_roll_pitch(dr["imu_tilt"][0], dr["imu_tilt"][1])
    noisy = noisy.at[L["gyro"]].set(rot @ noisy[L["gyro"]] + dr["gyro_bias"])
    noisy = noisy.at[L["gravity"]].set(rot @ noisy[L["gravity"]])
    noisy = noisy.at[L["joint_pos"]].add(dr["joint_offset"])

    lin_vel = gm.quat_rotate_inverse(quat, v[0:3])
    fh = self.foot_heights(d)
    priv = jnp.concatenate([
        clean, lin_vel, q[2:3], info["contact"].astype(jnp.float32), fh,
        info["feet_air_time"], d.actuator_force / self._torque_limit,
        self.dr.privileged(info["dr"], info["push_vel"])])
    return jnp.nan_to_num(noisy), jnp.nan_to_num(priv)

  def _fresh_episode(self, d, rng, dr_level):
    """New initial configuration written into an existing Data (no full forward)."""
    cfg = self.cfg
    ks = jax.random.split(rng, 10)
    nu = self.nu
    q = jnp.asarray(self._home_q)
    xy = jax.random.uniform(ks[0], (2,), minval=-cfg.init_xy_range, maxval=cfg.init_xy_range)
    yaw = jax.random.uniform(ks[1], (), minval=-cfg.init_yaw_range, maxval=cfg.init_yaw_range)
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
        # domain randomisation
        "dr_level": jnp.asarray(dr_level, jnp.float32),
        "dr": self.dr.sample(ks[8], dr_level),
        "push_vel": jnp.zeros(2),  # last applied kick, decaying (critic only)
    }
    info["push_timer"], info["push_kick"] = self.dr.sample_push(ks[9], dr_level)
    return d, info, ks[7]

  def _self_contacts(self, d):
    """Number of robot-robot contact pairs in penetration (legs, hands/arms vs torso and
    hips). MJX keeps one fixed-size contact slot per pair; slots of pairs that are far
    apart have a large positive distance."""
    c = getattr(d, "_impl", d).contact
    g = c.geom
    valid = (g[:, 0] >= 0) & (g[:, 1] >= 0)
    body = jnp.asarray(self._geom_body)
    b1, b2 = body[jnp.maximum(g[:, 0], 0)], body[jnp.maximum(g[:, 1], 0)]
    robot = (b1 > 0) & (b2 > 0) & ~jnp.asarray(self._ignore_pair)[b1, b2]
    return jnp.sum(valid & robot & (c.dist < 0)).astype(jnp.float32)

  def _reward_terms(self, d, d_prev, prev_info, phase, action, contact, first_contact,
                    air_time, fh, fell, self_contacts):
    """Build RewardInputs for one env and evaluate all reward terms."""
    q, v = d.qpos, d.qvel
    quat = q[3:7]
    heading = gm.yaw_quat(gm.quat_to_yaw(quat))
    feet = d.site_xpos[self._feet_site]
    feet_prev = d_prev.site_xpos[self._feet_site]
    rel = gm.quat_rotate_inverse(heading, feet[0] - feet[1])
    inp = rewards.RewardInputs(
        command=prev_info["command"],  # the command the action was chosen for
        lin_vel=gm.quat_rotate_inverse(heading, v[0:3]),
        ang_vel=v[3:6],
        gravity=gm.quat_rotate_inverse(quat, prev_info["dr"]["gravity_dir"]),
        base_height=q[2],
        joint_pos=q[self._qadr],
        joint_vel=v[self._vadr],
        joint_vel_prev=d_prev.qvel[self._vadr],
        torques=d.actuator_force,
        action=action,
        last_action=prev_info["last_act"],
        foot_height=fh,
        foot_vel_xy=(feet - feet_prev)[:, :2] / self.dt,
        foot_up_z=d.site_xmat[self._feet_site][:, 2, 2],
        contact=contact,
        first_contact=first_contact,
        air_time=air_time,
        phase=phase,
        feet_lateral_distance=jnp.abs(rel[1]),
        self_contacts=self_contacts,
        fell=fell,
    )
    return rewards.compute(inp, self._rconsts, self.cfg.reward)

  @staticmethod
  def _metrics(terms, ep_return, ep_len, fell, done):
    m = {f"reward/{k}": v for k, v in terms.items()}
    # fraction of steps where penalties outweigh rewards (sum clipped to 0)
    m["debug/reward_clipped"] = (rewards.pre_clip(terms) < 0).astype(jnp.float32)
    m.update({"episode/return": ep_return * done, "episode/length": ep_len * done,
              "episode/fell": fell * done, "episode/count": done})
    return m

  # ---------------------------------------------------------------- public API
  def refresh_obs(self, state: "State", rng) -> "State":
    """Recompute observations after editing `state.info` (e.g. an evaluation setting a
    command after reset), so the next action already sees the edit. Per env."""
    frame, priv = self._observe(self.packer.unpack(state.data), state.info, rng)
    return state._replace(obs=stack_frames(frame, None, self.history), priv_obs=priv)

  @staticmethod
  def set_dr_level(state: "State", level) -> "State":
    """Return `state` with a new randomisation level for all envs (batched or not).

    Takes effect for each env at its next reset (the current episode keeps its robot).
    """
    info = dict(state.info)
    info["dr_level"] = jnp.full_like(info["dr_level"], level)
    return state._replace(info=info)

  def reset(self, rng, dr_level=None) -> State:
    """New episode. `dr_level` (scalar) defaults to cfg.dr.level."""
    level = self.cfg.dr.level if dr_level is None else dr_level
    d = sim.init_data(self.mx, jnp.asarray(self._home_q), jnp.asarray(self._home_u))
    d, info, obs_rng = self._fresh_episode(d, rng, level)
    frame, priv = self._observe(d, info, obs_rng)
    obs = stack_frames(frame, None, self.history)
    zero = jnp.zeros(())
    terms = self._reward_terms(d, d, info, info["phase"], jnp.zeros(self.nu), info["contact"],
                               info["contact"] & False, info["feet_air_time"],
                               self.foot_heights(d), jnp.array(False), jnp.zeros(()))
    metrics = self._metrics({k: zero for k in terms}, zero, zero, zero, zero)
    metrics["debug/dr_level"] = zero
    metrics["debug/pushes"] = zero
    return State(self.packer.pack(d), obs, priv, zero, zero, zero, info, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    cfg = self.cfg
    info = dict(state.info)
    rng, k_obs, k_cmd, k_reset, k_push = jax.random.split(info["rng"], 5)
    d_prev = self.packer.unpack(state.data)
    model = self.dr.apply(self.mx, info["dr"])  # this env's randomised robot

    action = jnp.clip(action, -1.0, 1.0)
    # Latency: in "late" episodes the motors receive the action from 1 or 2 steps ago.
    lat = info["dr"]["latency"]
    applied = jnp.where(lat >= 2, info["last_last_act"], jnp.where(lat >= 1, info["last_act"], action))
    targets = jnp.clip(self._home_u + self.action_scale * applied, self._lo, self._hi)

    # Push: an instantaneous horizontal base-velocity kick every few seconds.
    info["push_timer"] = info["push_timer"] - 1
    push = info["push_timer"] <= 0
    kick = info["push_kick"] * push
    d_push = d_prev.replace(qvel=d_prev.qvel.at[0:2].add(kick))
    next_timer, next_kick = self.dr.sample_push(k_push, info["dr_level"])
    info["push_timer"] = jnp.where(push, next_timer, info["push_timer"])
    info["push_kick"] = jnp.where(push, next_kick, info["push_kick"])
    info["push_vel"] = info["push_vel"] * 0.9 + kick

    d = sim.step_ctrl(model, d_push, targets, self.n_substeps)

    # --- bookkeeping -------------------------------------------------------
    fh = self.foot_heights(d)
    contact = fh < cfg.contact_threshold
    prev_air = info["feet_air_time"]
    air_time = prev_air + self.dt  # time in the air including this step
    first_contact = (prev_air > 0) & contact
    info["feet_air_time"] = air_time * ~contact
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
    terms = self._reward_terms(d, d_prev, state.info, info["phase"], action, contact,
                               first_contact, air_time, fh, fell, self._self_contacts(d))
    reward = rewards.total(terms, cfg.reward)

    info["last_last_act"] = info["last_act"]
    info["last_act"] = action
    info["motor_targets"] = targets
    info["contact"] = contact
    info["ep_return"] = info["ep_return"] + reward
    info["rng"] = rng

    frame, priv = self._observe(d, info, k_obs)
    obs = stack_frames(frame, state.obs, self.history)

    f_done = done.astype(jnp.float32)
    metrics = self._metrics(terms, info["ep_return"], info["step"].astype(jnp.float32),
                            fell.astype(jnp.float32), f_done)
    metrics["debug/dr_level"] = info["dr_level"]
    metrics["debug/pushes"] = push.astype(jnp.float32)

    # --- auto-reset ------------------------------------------------------------
    d_new, info_new, k_obs2 = self._fresh_episode(d, k_reset, info["dr_level"])
    frame_new, priv_new = self._observe(d_new, info_new, k_obs2)
    obs_new = stack_frames(frame_new, None, self.history)
    d = sim.tree_where(done, d_new, d)
    info = sim.tree_where(done, info_new, info)
    obs = jnp.where(done, obs_new, obs)
    priv = jnp.where(done, priv_new, priv)

    return State(self.packer.pack(d), obs, priv, reward, f_done,
                 (timeout & ~fell).astype(jnp.float32), info, metrics)
