# g1_rl: Unitree G1 walking, end to end on GPU

A from-scratch reinforcement learning pipeline that teaches the Unitree G1 humanoid
(29 actuated joints, model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1))
to walk to velocity commands on flat ground.

- **Simulation:** MuJoCo **MJX** runs thousands of robots in parallel on the GPU.
- **Learning:** **custom PPO** in plain JAX, with no flax / optax / brax.
- **Hardware:** training targets Kaggle **2× T4**. Evaluation runs on GPU, with CPU as a fallback.

Training is entirely on device. The host only receives a handful of scalars per
iteration for logging.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Environment setup, MJX correctness check, throughput benchmark, time estimate | **done** |
| 1 | `G1Env` (reset/step/obs/termination, auto-reset, commands, gait clock) | **done** |
| 2 | Rewards (22 terms, unit-tested, balance check in sim) | **done** |
| 3 | Domain randomisation, action latency, pushes, curriculum level | **done** |
| 4 | PPO from scratch, multi-GPU train step, checkpoints/resume, policy export, logging, curriculum | **done** |
| 5 | Evaluation: MJX suite (GPU, CPU fallback), sim-to-sim in CPU MuJoCo incl. full mesh model, report, local viewer | **done** |
| 6 | Export: ONNX with embedded robot constants, portable `.npz`, standalone MuJoCo runner, numerical + closed-loop checks | **done** |

## Layout

```
assets/            MuJoCo Menagerie unitree_g1 (scene_mjx.xml = training scene)
g1_rl/
  config.py        configuration dataclasses
  device.py        GPU-by-default / CPU-fallback runtime + XLA flags + compile cache
  robot.py         model loading, joint groups, home pose (read from the MJCF)
  sim.py           MJX step helpers, DataPacker (see design notes)
  parallel.py      multi-GPU sharding helpers
  math.py          quaternion helpers
  env.py           G1Env: per-env reset/step (vmapped by callers)
  vec_env.py       batched multi-GPU wrapper + jitted rollouts
  rewards.py       reward terms: pure functions of RewardInputs (unit-tested)
  randomize.py     per-episode physics randomisation, latency, pushes
  render.py        offscreen MuJoCo video from qpos trajectories
  networks.py      MLPs + diagonal Gaussian policy (plain JAX)
  optim.py         Adam + global-norm clipping
  normalizer.py    running observation mean/variance
  ppo.py           PPO: GAE, losses, device-local minibatches, adaptive LR, fused train step
  trainer.py       training loop: resume, async metrics, checkpoints, time budget, video
  checkpoint.py    portable .npz/.json checkpoints and policy export
  policy.py        standalone loader to run an exported policy (GPU, CPU fallback)
  logger.py        compact console line + metrics.jsonl
  evaluate.py      MJX evaluation suite: one batched rollout over all scenarios
  sim2sim.py       independent CPU MuJoCo runner (training model + full mesh model)
  report.py        readiness checks, report.md/json, eval_summary.png
  export.py        ONNX export, NumPy reference, OnnxPolicy, deployment bundle
  symmetry.py      left-right mirror maps for joints / observations (mirror loss)
deploy/
  run_mujoco.py    standalone runner for policy.onnx (numpy + mujoco + onnxruntime only)
scripts/
  install_kaggle.sh  check_setup.py  benchmark.py  env_smoke.py  reward_check.py  dr_check.py
  train.py  plot_metrics.py  evaluate.py  play.py  render_traj.py  export.py
tests/             pytest suite (runs on GPU or CPU)
notebooks/kaggle_train.ipynb      interactive checks (each cell < 5 min)
notebooks/kaggle_overnight.ipynb  the overnight training run (Save & Run All)
```

## Quick start

**Kaggle (GPU T4 x2, Internet on):**
1. Upload this folder (or the zip) as a Kaggle Dataset, or push it to GitHub.
2. Open `notebooks/kaggle_train.ipynb` and run all cells.

**Local GPU (Linux/WSL2 with an NVIDIA driver):**

```bash
pip install -e ".[gpu,viz,dev]"
python scripts/check_setup.py              # GPU by default; falls back to CPU with a warning
python scripts/benchmark.py --scaling
python -m pytest -q
```

Use `--platform cpu` on any script to force CPU.

## Versions and GitHub

Each delivered package bumps `__version__` in `g1_rl/__init__.py`. On your
computer, `update_from_zip.sh -m "..." g1_rl_x_y_z.zip` commits it as one commit tagged
`v<version>` and pushes it. On Kaggle, the first notebook cell clones that tag
(`GIT_REF = "v1.0.0"`) or `main`. Run outputs (`outputs/`, checkpoints) are
git-ignored.

## Environment (Phase 1)

| | |
|---|---|
| Control | 50 Hz, 5 physics substeps of 4 ms. Action a ∈ [-1,1]^29 → PD target `home + s·a` (clipped to joint range); s = 0.5 rad for legs and waist, 0.25 rad for arms (v0.8, `action_scale_overrides`) |
| Policy obs (5 × 100) | the last 5 frames, newest first (v0.9). A frame: gyro, gravity in body frame, command (vx, vy, ωz), joint pos − home, joint vel, last action, gait clock cos/sin (2 feet). Uniform noise plus this episode's sensor errors (IMU tilt, gyro bias, encoder offsets) |
| Critic obs (212) | clean current frame + base linear velocity, pelvis height, foot contact, foot height, foot air time, actuator torque / limit, this episode's randomisation (friction, payload, CoM offset, kp/kd scales, latency, recent push, slope, IMU tilt, gyro bias) |
| Commands | vx ∈ [−0.6, 1.0] m/s, vy ∈ [−0.5, 0.5] m/s, ωz ∈ [−1, 1] rad/s; 20 % stand-still, 20 % small (\|vx\|,\|vy\| ≤ 0.3, \|ωz\| ≤ 0.4), deadband 0.05; resampled every 5–10 s |
| Reset | `knees_bent` pose + joint noise, random heading and xy, small base velocity; lowest foot placed on the floor |
| Termination | pelvis < 0.5 m, tilt > 60°, or non-finite state. Time limit 20 s (flagged as truncation, not failure) |
| Foot contact | analytic: lowest point of each foot's 3 collision capsules above the flat floor |

`env.obs_layout` (one frame; the current frame is `obs[:100]`) / `env.priv_layout` give named slices into the observation vectors.

## Rewards (Phase 2)

Per step: `r = max(Σ wᵢ·termᵢ·dt, 0) + w_term·fell·dt` (dt = 0.02 s). Weights live in
`config.RewardConfig.weights`; weight 0 disables a term. Logged per term as
`reward/<name>` (already weighted and dt-scaled, so the values are shares of the return).

| Group | Term | Weight | What it measures |
|---|---|---|---|
| Task | tracking_lin_vel | 1.5 | exp(−‖v_xy,cmd − v_xy‖²/0.25), velocity in the heading frame |
| | tracking_ang_vel | 1.0 | exp(−(ω_z,cmd − ω_z)²/0.25) (0.75 before v0.7) |
| | tracking_lin_vel_fine / tracking_ang_vel_fine | 0.5 / 0.5 | the same errors with σ 0.05: small errors and ignored small commands cost reward (v0.9.1) |
| | alive | 0.25 | 1 every step |
| Gait | feet_phase | 1.0 | swing-foot height follows the clock (peak 8 cm); both feet down when standing (*) |
| | gait_contact | 0.5 | contact pattern matches the clock |
| | feet_air_time | 1.0 | on touchdown: min(air − 0.3 s, 0.2 s); negative for shuffling steps; only while moving |
| | feet_slip | −0.25 | ‖v_xy‖² of feet in contact |
| | feet_flat | −2 | 1 − cos(sole tilt) for feet in contact |
| | feet_distance | −30 | feet closer than 15 cm sideways (they stand 24 cm apart; −10 before v0.9) |
| | self_collision | −1 | number of robot–robot contact pairs in penetration: legs, hands vs hips, arms vs torso (v0.9) |
| Base | lin_vel_z / ang_vel_xy / orientation / base_height | −1 / −0.05 / −2 / −10 | bouncing, wobbling, leaning, pelvis away from 0.74 m |
| Posture | pose | −0.5 | per-joint weighted (q − q_home)²: free hips/knees, light arm swing (0.2), stiff hip roll/yaw, waist, wrists |
| | stand_still | −0.2 | Σ\|q − q_home\| when standing (*) |
| | arm_still | −1 | Σ (q − q_home)² over shoulder/elbow/wrist joints when standing (*) (v0.8) |
| | joint_limits | −5 | beyond 95 % of each joint range |
| Effort | joint_vel / joint_acc / torques / energy / action_rate | −1e−3 / −2.5e−7 / −1e−5 / −1e−4 / −0.05 | regularisation |
| Failure | termination | −50 | on a fall, added after the clip (−1.0 per fall) |

(*) "Standing" = zero command **and** base slower than 0.25 m/s (`stand_recovery_speed`,
v0.7). If a standing robot is pushed and starts sliding, the feet follow the gait clock
and stand_still is off, so a recovery step is rewarded instead of penalised. (Measured
over 24 pushes per strength this changed little: g1_walk_v1 already stepped after
pushes and survived 0.5 m/s standing pushes 96 % of the time.)

The reward uses the command the action was chosen for, so a command change does not
mis-credit the step on which it happens.

## Domain randomisation (Phase 3)

Sampled once per episode (`config.DomainRandConfig`), applied every step by writing the
values into this env's copy of the MJX model, so under vmap every robot is different.

| Parameter | Range (level 1) | Nominal |
|---|---|---|
| Floor friction (all floor contacts) | 0.3 – 1.2 | 1.0 |
| Link masses (inertia scaled with them) | ×0.9 – 1.1 per link | ×1 |
| Torso payload | −1 – +8 kg | 0 |
| Torso centre-of-mass offset | ±3 cm per axis | 0 |
| Motor stiffness kp / damping kd | ×0.75 – 1.25 / ×0.7 – 1.3 per motor | ×1 |
| Joint friction / armature | ×0.5 – 2.0 / ×0.95 – 1.05 per joint | ×1 |
| Action latency | 40 % of episodes 1 step (20 ms) late, 20 % 2 steps (40 ms) | none |
| Pushes | 0.1 – 1.0 m/s horizontal kick every 5 – 10 s | none |
| Slope (v0.9) | 60 % of episodes on a slope of 0 – 10° in a random direction, simulated by tilting gravity | flat |
| IMU mounting tilt / gyro bias (v0.9) | ±2.5° roll and pitch / ±0.1 rad/s per axis (policy obs only) | none |
| Joint-encoder offset (v0.9) | ±0.03 rad per joint (policy obs only) | none |

`level` ∈ [0, 1] (`info["dr_level"]`) shrinks every range toward nominal;
`G1Env.set_dr_level(state, x)` changes it for all envs, effective at each env's next
reset. Phase 4 uses it as a curriculum. The critic observes the sampled values; the
policy does not and must be robust to them.

## Training (Phase 4)

```bash
python scripts/train.py --preset smoke                       # < 5 min check
python scripts/train.py --preset full --max-hours 11         # ~5.5-6 h on Kaggle 2xT4
python scripts/train.py --preset full --resume outputs/runs/<run>   # continue
python scripts/plot_metrics.py outputs/runs/<run> --show-terms
```

| | |
|---|---|
| Algorithm | PPO, asymmetric actor-critic (policy sees 5 × 100 obs, critic 212), GAE (γ 0.99, λ 0.95), clipped surrogate + clipped value loss, entropy 0.002 (0.005 before v0.9.1), action std ≤ 1.0, grad-norm 1.0 |
| Batch | 8192 envs × 24 steps = 196,608 steps / iteration; 5 epochs × 4 minibatches; 2035 iterations = 400M steps (250M before v0.9) |
| Learning rate | Adam, 1e-3 initial, adapted every minibatch to keep KL ≈ 0.01 (rsl_rl rule) |
| Networks | 512-256-128 ELU for both; learned state-independent action std (init 0.6) |
| Normalisation | running mean/var of both observation vectors, clipped at ±10; the loss uses the statistics the rollout used, updated afterwards |
| Multi-GPU | envs split across GPUs; minibatches drawn within each GPU's samples (no shuffling traffic); gradients all-reduced |
| Curriculum | randomisation level (incl. slopes and pushes) starts at 0 and rises by 0.02 per iteration while velocity tracking ≥ 60 % of max and falls are rarer than 1 per 800 steps |
| Time-outs | 20 s time limits bootstrap from V(s) (not treated as failures) |
| Bound loss (v0.8) | 0.5 · Σ relu(\|μ\| − 1)² : the env clips actions to [−1, 1], so without it a mean that drifts past a bound gets no gradient back and stays saturated (logged: `loss/bound`, `policy/saturated`) |
| Mirror loss (v0.8) | 1.0 · ‖μ(M_o·o) − M_a·μ(o)‖²: the policy must act on the left-right mirrored observation with the mirrored action (`symmetry.py`; roll/yaw joints and lateral quantities change sign, left/right swap, the feet swap gait clocks). Gives symmetric gaits; logged as `loss/symmetry` |

Outputs in `outputs/runs/<run>/`: `metrics.jsonl` (every iteration), `ckpt/` (last 3
full states), `policy_latest.npz`/`policy_final.npz` (+ `.json`), `video_walk_forward.mp4`,
`training_curves.png`, `config.json`, `summary.json`.

**Running a trained policy on your own machine** (only jax + numpy needed):

```python
from g1_rl.policy import Policy
pi = Policy.load("policy_final.npz")       # GPU if available, else CPU
targets = pi.joint_targets(obs)             # obs: 100-vector, layout in policy_final.json
```

## Evaluation (Phase 5)

```bash
python scripts/evaluate.py --run outputs/runs/<run>            # ~5 min on Kaggle 2xT4; --platform cpu works too
python scripts/play.py --policy outputs/runs/<run>/policy_final.npz   # local, keyboard-driven viewer
```

**MJX suite** (GPU by default, CPU fallback): every scenario is one env in a single batch.

| Group | Scenarios | Measures |
|---|---|---|
| grid | nominal robot; vx ∈ {−0.5…1.0}, vy ∈ {−0.4, 0, 0.4}, ωz ∈ {−0.8, 0, 0.8} | survival, velocity / yaw-rate tracking error, cost of transport, slip, step frequency, swing height, torque and joint-limit margins |
| training | full randomisation, random commands, pushes (as in training) | survival, falls per minute, tracking |
| ood | one parameter beyond its training range (friction 0.2–1.2, payload −2–8 kg, kp ×0.6–1.4, kd ×0.5–2, CoM ±6 cm, joint friction ×0.5–4, latency) | survival and tracking per value |
| push | standing / walking 0.5 m/s, single kick 0.25–2 m/s, 4 directions | largest push always survived |

**Sim-to-sim** (`sim2sim.py`) runs the exported policy in classic CPU MuJoCo (float64), with its
own observation builder and action mapping taken only from the policy file. That is the same
information a deployment controller has. It is verified against the training env: identical
observations, and matching closed-loop motion for 0.5 s. It runs 9 scenarios on the training
model and on the full mesh model (`scene.xml`: mesh contacts, 2 ms steps, 100 solver
iterations, joint friction 3× training nominal; PD gains set to the policy's).

Output: `report.md` with ✅/❌ readiness checks and tables, `report.json`, `eval_summary.png`,
videos (`mjx_forward_0p5.mp4`, `full_model_forward_0p5.mp4`, `full_model_walk_push_left_0p5.mp4`).

## Policy refinements after the first runs (v0.8)

g1_walk_v1 and v2 held the right arm back and straight while standing (shoulder pitch and
elbow outputs pinned at +1, the action limit). Causes: the arm posture weight was low, the
clipped action gave no gradient back once the mean passed +1, and nothing asked for
left-right symmetry (both runs used seed 0, so the same quirk). v0.8 adds the bound loss,
the mirror loss, a 0.25 rad action range for the arms and the `arm_still` reward.

## Changes from the v3 test campaign (v0.9)

A 2,450-episode CPU MuJoCo test of g1_walk_v3 found four main weak spots; v0.9 targets each:

| Finding in v3 | Change |
|---|---|
| No sense of its own velocity: walked off slopes and with a tilted IMU (0.35–0.45 m/s while told to stand) | 5-frame observation history; 60 % of episodes on 0–10° slopes; IMU tilt, gyro bias and encoder offsets randomised |
| Ignored commands below ~0.2 m/s and ~0.3 rad/s | sharp tracking terms (σ 0.05) beside the broad ones, 20 % small commands, deadband 0.05 |
| Legs collided when stepping sideways; weak against side pushes while standing | self_collision penalty, feet_distance −30, 20 % standing, pushes to 1.0 m/s |
| Fell at kp ×0.7, payload 12 kg; 4-step latency | kp ×0.75–1.25, payload to 8 kg, latency up to 2 steps, friction 0.3–1.2 |

**v0.9.1:** v0.9.0 replaced the broad tracking kernels with σ 0.1. The positive reward shrank
so much that the entropy bonus won: the action std rose from 0.4 to 0.8, falls in training
stayed at 50–60 % for 1,000 iterations, and g1_walk_v4 turned at only ~35 % of the commanded
rate. v0.9.1 restores σ 0.25, adds the sharp terms, lowers the entropy bonus to 0.002 and
caps the action std at 1.0. Everything else from v0.9.0 stays (v4 already handled 9° slopes
and 1–2 m/s standing pushes).

Not yet included: real uneven terrain (steps, rough ground). MJX heightfields need the
analytic foot-contact code rewritten, so this stays for a later version; slopes are covered
exactly by the gravity tilt. The evaluation suite gained slope, IMU-tilt and small-command
groups with readiness checks.

## Final version (v1.0): lessons from v1–v4

g1_walk_v4 (v0.9.0) was tested on CPU MuJoCo against v3. It is far more robust:
- standing push survival at 1.0 m/s rose from 33 % to 94 %
- 9° slopes are survived walking and standing
- self-contacts fell from 37 % to 0
- it handles a 15 kg payload and kp ×0.7

It lost the commands, though:
- it turns at only ~35 % of the commanded yaw rate (v3: 99 %)
- top speed is 0.77 m/s at a 1.0 m/s command (v3: 0.96)
- 17 % of its actions sit at the ±1 limit (ankle roll 70–90 %, shoulder roll 54–85 %)

What the tests showed, and what v1.0 does about it:

| Test | Result | v1.0 |
|---|---|---|
| Score v3's and v4's trajectories under both reward sets | Both reward sets rate v3's behaviour ~20 % higher than v4's, so v4 did not find a better optimum. It got stuck because the σ 0.1 kernel gives almost no gradient at large errors, and the entropy bonus pushed its std from 0.4 to 0.8. | broad σ 0.25 kernels plus sharp σ 0.05 terms; entropy 0.002; std cap 1.0 (v0.9.1) |
| Seven CPU fine-tunes of v4 (warm start, critic warm-up, lr 1e-4) | Training yaw tracking stayed flat at 0.18 in every run, and 100 iterations made speed and small-command tracking slightly worse under either bound setting. Starting at lr 1e-3 jumped to KL 0.37 on the first update. | **train from scratch**. The warm start (`--init-policy`, `--critic-warmup`) stays available, but there is no evidence that it repairs turning. |
| Ankle-roll action scale 0.25 | Saturation 19 % → 7 %, but top speed −18 % | keep 0.5 (v3 had 2 % saturation at 0.5) |
| Bound loss 2.0 vs 0.5 | Saturation 1 % vs 8 %; tracking fell in both | 1.0 |
| Hand capsules vs hips/thighs | Penalised in v4's training model; makes it hold its arms spread at the limit. The full mesh model shows almost no such contact. | these pairs no longer count for `self_collision`; leg–leg contacts still do |

The evaluation report adds a check that at most 5 % of actions are pinned at ±1.
`pretrained/g1_walk_v4/` ships v4's policy as a fallback.

Run: `notebooks/kaggle_overnight.ipynb` with `RUN_NAME = "g1_walk_final"`, `MAX_HOURS = 7.2`
(350M steps take ~7.1 h; ~7.5 h in total including evaluation and export).

## Export (Phase 6)

```bash
python scripts/export.py --run outputs/runs/<run>        # -> outputs/export/<run>/, < 1 min on CPU
```

| file | |
|---|---|
| `policy.onnx` | input `obs` (batch, 100) → outputs `action` ∈ [−1, 1] and `joint_targets` (rad, PD targets). Normaliser, clips and action mapping are in the graph; observation layout, joint order, home pose, kp/kd, control rate are in the model metadata (`g1_rl` key), so this file alone is enough |
| `policy.npz/.json` | the same policy for `g1_rl.policy.Policy` (JAX) |
| `run_mujoco.py` | standalone runner/reference controller: `numpy`, `mujoco`, `onnxruntime` only; `--viewer` for keyboard control |
| `README.md`, `export_info.json` | interface description; check results |

Checks: ONNX = NumPy reference = JAX on 4096 observations (max difference ~1e-5), then the ONNX
policy walks 10 s in the full mesh model, through both the g1_rl sim-to-sim runner and the
standalone script. `--include-model` copies the Menagerie G1 model into the bundle.
`g1_rl.export.OnnxPolicy` has the `Policy` interface, so `scripts/play.py --policy x.onnx` works too.

## Design notes (Phase 0 findings)

- **The MJX model matches regular MuJoCo.** `scene_mjx.xml` stepped in MJX (float32) tracks the
  CPU MuJoCo reference (float64) to <1e-3 rad over 100 physics steps from the home pose.
- **Zero-size arrays are kept off the device boundary.** `mjx.Data` for this robot contains zero-size arrays. Every array crossing a
  jit boundary is therefore packed with `sim.DataPacker` (non-empty leaves only) and
  unpacked inside the traced function.
- **Multi-GPU uses sharded jit, not shard_map.** Inputs and outputs are sharded along the env axis
  (`parallel.jit_batched`), and XLA partitions the work. `shard_map`/`pmap` around
  MJX's forward pass segfaulted XLA in multi-device runs (reproduced on CPU with
  jax 0.11.2 / mujoco 3.16.0), so it is only an opt-in benchmark mode.
  - The one cross-device op in the simulator is the constraint solver's 1-byte
    "any env still iterating?" flag.
  - PPO gradients will be averaged by the same mechanism.
- **Cheap auto-reset.** A finished episode is replaced in the same step using only `mjx.kinematics`
  (no full forward pass); `mjx.step` recomputes all dynamics from qpos/qvel next step.
- **No silent recompiles.** Inputs to jitted functions must already be sharded (`VecEnv.keys()`),
  otherwise JAX recompiles on the second call; the smoke test checks this.
- **Checkpoints will be portable.** They will be plain NumPy `.npz` files plus a JSON config, so a
  policy trained on Kaggle loads on any machine with only `jax` + `numpy`.
