"""Phase 2: reward terms on hand-made inputs (no simulation, runs in seconds).

Baseline = a robot doing exactly what is asked: tracking the command, upright at
the target height, home pose, feet following the gait clock. Every penalty must
be exactly 0 there and every task/gait reward at its maximum; then each test
perturbs one quantity and checks the matching term reacts (and only sensibly).
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from g1_rl import rewards, robot
from g1_rl.config import RewardConfig, SimConfig

CFG = RewardConfig()
DT = SimConfig().ctrl_dt


@pytest.fixture(scope="module")
def consts():
  m = robot.load_mj_model()
  info = robot.robot_info(m)
  return rewards.make_consts(info.actuator_names, info.home_ctrl, info.ctrl_lo, info.ctrl_hi, CFG, DT)


def baseline(c, cmd=(0.5, 0.0, 0.0), phase=(np.pi / 2, -np.pi / 2)):
  """Left foot mid-swing (at peak height), right foot in stance."""
  nu = len(c.home)
  cmd = jnp.asarray(cmd, jnp.float32)
  moving = float(np.linalg.norm(cmd) > 0)
  phase = jnp.asarray(phase, jnp.float32)
  fh = CFG.swing_height * jnp.maximum(jnp.sin(phase), 0) * moving
  contact = fh <= 0
  return rewards.RewardInputs(
      command=cmd, lin_vel=jnp.array([cmd[0], cmd[1], 0.0]), ang_vel=jnp.array([0.0, 0.0, cmd[2]]),
      gravity=jnp.array([0.0, 0.0, -1.0]), base_height=jnp.asarray(CFG.base_height_target),
      joint_pos=jnp.asarray(c.home), joint_vel=jnp.zeros(nu), joint_vel_prev=jnp.zeros(nu),
      torques=jnp.zeros(nu), action=jnp.zeros(nu), last_action=jnp.zeros(nu),
      foot_height=fh, foot_vel_xy=jnp.zeros((2, 2)), foot_up_z=jnp.ones(2),
      contact=contact, first_contact=jnp.zeros(2, bool), air_time=jnp.zeros(2), phase=phase,
      feet_lateral_distance=jnp.asarray(0.24), fell=jnp.asarray(False))


def terms_of(i, c, cfg=CFG):
  return {k: float(v) for k, v in rewards.compute(i, c, cfg).items()}


W = CFG.weights
REWARDS = ("tracking_lin_vel", "tracking_ang_vel", "alive", "feet_phase", "gait_contact")


def test_pose_weights_cover_every_joint_once(consts):
  names = robot.robot_info(robot.load_mj_model()).actuator_names
  for n in names:
    matches = [k for k in CFG.pose_weights if k in n]
    assert len(matches) == 1, (n, matches)
  assert (consts.soft_lo < consts.home).all() and (consts.home < consts.soft_hi).all()


@pytest.mark.parametrize("cmd", [(0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.3, 0.4, 0.8)])
def test_baseline_is_optimal(consts, cmd):
  t = terms_of(baseline(consts, cmd), consts)
  for k in REWARDS:
    assert t[k] == pytest.approx(W[k] * DT, rel=1e-5), k
  for k, v in t.items():
    if k not in REWARDS:
      assert v == pytest.approx(0.0, abs=1e-9), (k, v)


def test_tracking_falls_off_with_error(consts):
  i = baseline(consts)
  good = terms_of(i, consts)["tracking_lin_vel"]
  half = terms_of(i._replace(lin_vel=jnp.array([0.0, 0.0, 0.0])), consts)["tracking_lin_vel"]
  assert half == pytest.approx(good * np.exp(-0.25 / CFG.tracking_sigma), rel=1e-5)
  spin = terms_of(i._replace(ang_vel=jnp.array([0.0, 0.0, 1.0])), consts)["tracking_ang_vel"]
  assert spin < 0.1 * W["tracking_ang_vel"] * DT


def test_base_penalties(consts):
  i = baseline(consts)
  t = terms_of(i._replace(lin_vel=i.lin_vel.at[2].set(0.5)), consts)
  assert t["lin_vel_z"] == pytest.approx(W["lin_vel_z"] * 0.25 * DT)
  t = terms_of(i._replace(gravity=jnp.array([0.0, np.sin(0.3), -np.cos(0.3)])), consts)
  assert t["orientation"] == pytest.approx(W["orientation"] * np.sin(0.3) ** 2 * DT, rel=1e-5)
  t = terms_of(i._replace(base_height=jnp.asarray(CFG.base_height_target - 0.1)), consts)
  assert t["base_height"] == pytest.approx(W["base_height"] * 0.01 * DT, rel=1e-4)
  t = terms_of(i._replace(ang_vel=jnp.array([1.0, 1.0, 0.0])), consts)
  assert t["ang_vel_xy"] == pytest.approx(W["ang_vel_xy"] * 2 * DT)


def test_pose_respects_per_joint_weights(consts):
  i = baseline(consts)
  names = robot.robot_info(robot.load_mj_model()).actuator_names
  knee, hip_roll = names.index("left_knee_joint"), names.index("left_hip_roll_joint")
  t_knee = terms_of(i._replace(joint_pos=i.joint_pos.at[knee].add(0.3)), consts)["pose"]
  t_roll = terms_of(i._replace(joint_pos=i.joint_pos.at[hip_roll].add(0.3)), consts)["pose"]
  assert t_roll < t_knee < 0  # hip roll deviation costs far more than knee bend


def test_stand_still_only_when_command_is_zero(consts):
  for cmd, expect_zero in [((0.5, 0, 0), True), ((0, 0, 0), False)]:
    i = baseline(consts, cmd)
    t = terms_of(i._replace(joint_pos=i.joint_pos + 0.1), consts)
    assert (t["stand_still"] == 0) == expect_zero


def test_joint_limits(consts):
  i = baseline(consts)
  t = terms_of(i._replace(joint_pos=jnp.asarray(consts.soft_hi) + 0.1), consts)
  assert t["joint_limits"] == pytest.approx(W["joint_limits"] * 0.1 * len(consts.home) * DT, rel=1e-4)
  t = terms_of(i._replace(joint_pos=jnp.asarray(consts.soft_hi) - 1e-3), consts)
  assert t["joint_limits"] == 0


def test_effort_and_smoothness(consts):
  i = baseline(consts)
  nu = len(consts.home)
  t = terms_of(i._replace(torques=jnp.full(nu, 10.0), joint_vel=jnp.full(nu, 2.0)), consts)
  assert t["torques"] == pytest.approx(W["torques"] * 100 * nu * DT, rel=1e-5)
  assert t["energy"] == pytest.approx(W["energy"] * 20 * nu * DT, rel=1e-5)
  assert t["joint_vel"] == pytest.approx(W["joint_vel"] * 4 * nu * DT, rel=1e-5)
  assert t["joint_acc"] == pytest.approx(W["joint_acc"] * (2 / DT) ** 2 * nu * DT, rel=1e-4)
  t = terms_of(i._replace(action=jnp.full(nu, 0.5)), consts)
  assert t["action_rate"] == pytest.approx(W["action_rate"] * 0.25 * nu * DT, rel=1e-5)


def test_feet_slip_only_counts_feet_in_contact(consts):
  i = baseline(consts)  # left foot swinging, right in contact
  v = jnp.array([[1.0, 0.0], [0.0, 0.0]])
  assert terms_of(i._replace(foot_vel_xy=v), consts)["feet_slip"] == 0  # swing foot moving: fine
  t = terms_of(i._replace(foot_vel_xy=v[::-1]), consts)["feet_slip"]
  assert t == pytest.approx(W["feet_slip"] * DT)


def test_feet_flat_and_distance(consts):
  i = baseline(consts)
  t = terms_of(i._replace(foot_up_z=jnp.array([1.0, np.cos(0.2)])), consts)  # stance foot tilted
  assert t["feet_flat"] == pytest.approx(W["feet_flat"] * (1 - np.cos(0.2)) * DT, rel=1e-4)
  t = terms_of(i._replace(feet_lateral_distance=jnp.asarray(0.05)), consts)
  assert t["feet_distance"] == pytest.approx(W["feet_distance"] * 0.10 * DT, rel=1e-4)


def test_air_time_on_touchdown_only(consts):
  i = baseline(consts)
  long_step = i._replace(first_contact=jnp.array([True, False]), air_time=jnp.array([0.45, 0.0]))
  assert terms_of(long_step, consts)["feet_air_time"] == pytest.approx(
      W["feet_air_time"] * 0.15 * DT, rel=1e-4)
  short_step = long_step._replace(air_time=jnp.array([0.1, 0.0]))
  assert terms_of(short_step, consts)["feet_air_time"] < 0  # shuffling is penalised
  huge = long_step._replace(air_time=jnp.array([2.0, 0.0]))  # capped
  assert terms_of(huge, consts)["feet_air_time"] == pytest.approx(
      W["feet_air_time"] * CFG.air_time_max_bonus * DT, rel=1e-4)
  no_touch = long_step._replace(first_contact=jnp.array([False, False]))
  assert terms_of(no_touch, consts)["feet_air_time"] == 0
  standing = baseline(consts, (0, 0, 0))._replace(first_contact=jnp.array([True, False]),
                                                  air_time=jnp.array([0.45, 0.0]))
  assert terms_of(standing, consts)["feet_air_time"] == 0  # no stepping reward when told to stand


def test_gait_terms_follow_clock(consts):
  i = baseline(consts)
  # Feet swapped relative to the clock: both gait terms drop.
  wrong = i._replace(foot_height=i.foot_height[::-1], contact=i.contact[::-1])
  t_ok, t_bad = terms_of(i, consts), terms_of(wrong, consts)
  assert t_bad["gait_contact"] == 0
  assert t_bad["feet_phase"] < 0.1 * t_ok["feet_phase"]
  # Standing command: both feet down is optimal regardless of the clock.
  s = baseline(consts, (0, 0, 0))
  assert bool(s.contact.all())
  assert terms_of(s, consts)["feet_phase"] == pytest.approx(W["feet_phase"] * DT)


def test_total_clip_and_termination(consts):
  i = baseline(consts)
  bad = i._replace(gravity=jnp.array([0.0, 1.0, 0.0]), base_height=jnp.asarray(0.3),
                   lin_vel=jnp.array([0.0, 0.0, -2.0]))
  t = rewards.compute(bad, consts, CFG)
  assert float(sum(v for k, v in t.items() if k != "termination")) < 0
  assert float(rewards.total(t, CFG)) == 0.0  # clipped at zero
  fell = rewards.compute(bad._replace(fell=jnp.asarray(True)), consts, CFG)
  assert float(rewards.total(fell, CFG)) == pytest.approx(W["termination"] * DT)  # after the clip
  no_clip = dataclasses.replace(CFG, only_positive=False)
  assert float(rewards.total(t, no_clip)) < 0


def test_weight_config(consts):
  w = dict(CFG.weights, energy=0.0)
  t = rewards.compute(baseline(consts), consts, dataclasses.replace(CFG, weights=w))
  assert "energy" not in t  # weight 0 disables a term
  with pytest.raises(KeyError):
    rewards.compute(baseline(consts), consts, dataclasses.replace(CFG, weights={"typo": 1.0}))


def test_pushed_while_standing_may_step(consts):
  """Zero command but the base is sliding (pushed): the feet should follow the clock
  (a recovery step is rewarded, not penalised) and stand_still is off. Once the base
  is slow again, the stand-still rewards are back."""
  thr = CFG.stand_recovery_speed
  s = baseline(consts, (0, 0, 0))
  pushed = s._replace(lin_vel=jnp.array([thr + 0.2, 0.0, 0.0]))
  step_h = CFG.swing_height * jnp.maximum(jnp.sin(s.phase), 0)  # left foot up, as the clock says
  stepping = pushed._replace(foot_height=step_h, contact=step_h <= 0,
                             joint_pos=s.joint_pos + 0.1)
  feet_down = pushed._replace(joint_pos=s.joint_pos + 0.1)
  t_step, t_down = terms_of(stepping, consts), terms_of(feet_down, consts)
  assert t_step["feet_phase"] == pytest.approx(W["feet_phase"] * DT)
  assert t_step["gait_contact"] == pytest.approx(W["gait_contact"] * DT)
  assert t_down["feet_phase"] < t_step["feet_phase"] and t_down["gait_contact"] < t_step["gait_contact"]
  assert t_step["stand_still"] == 0
  assert t_step["feet_air_time"] == 0  # no stepping *bonus* for a standing command
  slow = s._replace(lin_vel=jnp.array([thr - 0.1, 0.0, 0.0]), joint_pos=s.joint_pos + 0.1)
  t_slow = terms_of(slow, consts)
  assert t_slow["stand_still"] < 0 and t_slow["feet_phase"] == pytest.approx(W["feet_phase"] * DT)
