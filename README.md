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
| 2 | Rewards | next |
| 3 | Domain randomisation + pushes | |
| 4 | PPO training, checkpoints, resume, logging | |
| 5 | Evaluation (GPU MJX by default, CPU MuJoCo fallback + sim-to-sim) | |
| 6 | Export (portable `.npz` + ONNX) | |

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
  rewards.py       reward terms (Phase 1: alive placeholder)
  render.py        offscreen MuJoCo video from qpos trajectories
scripts/
  install_kaggle.sh  check_setup.py  benchmark.py  env_smoke.py  render_traj.py
tests/             pytest suite (runs on GPU or CPU)
notebooks/kaggle_train.ipynb
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
computer, `update_from_zip.sh g1_rl.zip` commits it as one commit tagged
`v<version>` and pushes it. On Kaggle, the first notebook cell clones that tag
(`GIT_REF = "v0.2.2"`) or `main`. Run outputs (`outputs/`, checkpoints) are
git-ignored.

## Environment (Phase 1)

| | |
|---|---|
| Control | 50 Hz, 5 physics substeps of 4 ms. Action a ∈ [-1,1]^29 → PD target `home + 0.5·a` (clipped to joint range) |
| Policy obs (100) | gyro, gravity in body frame, command (vx, vy, ωz), joint pos − home, joint vel, last action, gait clock cos/sin (2 feet). Uniform noise |
| Critic obs (139) | clean policy obs + base linear velocity, pelvis height, foot contact, foot height, foot air time, actuator torque / limit |
| Commands | vx ∈ [−0.6, 1.0] m/s, vy ∈ [−0.5, 0.5] m/s, ωz ∈ [−1, 1] rad/s; 10 % stand-still; resampled every 5–10 s |
| Reset | `knees_bent` pose + joint noise, random heading and xy, small base velocity; lowest foot placed on the floor |
| Termination | pelvis < 0.5 m, tilt > 60°, or non-finite state. Time limit 20 s (flagged as truncation, not failure) |
| Foot contact | analytic: lowest point of each foot's 3 collision capsules above the flat floor |

`env.obs_layout` / `env.priv_layout` give named slices into both observation vectors.

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
