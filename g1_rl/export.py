"""Phase 6: export a trained policy for deployment.

Three equivalent forms of the same deterministic controller (obs -> action):

  policy.npz + policy.json   the training export (weights, normaliser, robot constants)
  policy.onnx                a self-contained ONNX graph; the JSON metadata (observation
                             layout, joint order, home pose, PD gains, ...) is embedded
                             in the model's metadata_props["g1_rl"], so this one file is
                             enough to drive the robot
  numpy_forward()            a 10-line NumPy reference used to check the other two

ONNX graph (opset 17), input "obs" (batch, 100) float32, two outputs:
  action        = clip(MLP(clip((obs - mean) / std, -c, c)), -1, 1)
  joint_targets = min(max(home + action_scale * action, ctrl_lo), ctrl_hi)   # PD targets, rad

`OnnxPolicy` runs the ONNX file with onnxruntime (CUDA if available, else CPU) and has
the same interface as `g1_rl.policy.Policy`, so the sim-to-sim runner and the viewer
accept either.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

META_KEY = "g1_rl"
ROBOT_ARRAYS = ("home_ctrl", "ctrl_lo", "ctrl_hi", "kp", "kd")


# ----------------------------------------------------------------------------- reference
def _act(name, x):
  if name == "elu":
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
  if name == "relu":
    return np.maximum(x, 0)
  if name == "tanh":
    return np.tanh(x)
  if name == "silu":
    return x / (1 + np.exp(-x))
  raise ValueError(f"unknown activation {name!r}")


def numpy_forward(arrays: dict, meta: dict, obs) -> np.ndarray:
  """Deterministic action in [-1, 1], computed in float64 with NumPy only."""
  x = np.asarray(obs, np.float64)
  std = np.sqrt(np.asarray(arrays["obs_var"], np.float64) + 1e-8)
  c = float(meta["obs_clip"])
  x = np.clip((x - arrays["obs_mean"]) / std, -c, c)
  n = int(meta["num_layers"])
  for i in range(n):
    x = x @ np.asarray(arrays[f"layer{i}.w"], np.float64) + arrays[f"layer{i}.b"]
    if i < n - 1:
      x = _act(meta["activation"], x)
  return np.clip(x, -1.0, 1.0)


def load_npz(path: str):
  with np.load(path) as z:
    arrays = {k: z[k] for k in z.files}
  meta_path = path[:-4] + ".json"
  if not os.path.exists(meta_path):
    raise FileNotFoundError(f"policy metadata {meta_path} not found next to {path}")
  return arrays, json.load(open(meta_path))


# ----------------------------------------------------------------------------- ONNX
def to_onnx(arrays: dict, meta: dict, path: str, opset: int = 17) -> str:
  import onnx  # noqa: PLC0415
  from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

  f32 = lambda a: np.asarray(a, np.float32)
  inits, nodes = [], []

  def const(name, value):
    inits.append(numpy_helper.from_array(f32(value), name))
    return name

  act_op = {"elu": "Elu", "relu": "Relu", "tanh": "Tanh"}.get(meta["activation"])
  if act_op is None and meta["activation"] != "silu":
    raise ValueError(f"activation {meta['activation']!r} not supported for ONNX export")

  c = float(meta["obs_clip"])
  inv_std = 1.0 / np.sqrt(np.asarray(arrays["obs_var"], np.float64) + 1e-8)
  nodes += [
      helper.make_node("Sub", ["obs", const("obs_mean", arrays["obs_mean"])], ["x_centered"]),
      helper.make_node("Mul", ["x_centered", const("obs_inv_std", inv_std)], ["x_scaled"]),
      helper.make_node("Clip", ["x_scaled", const("obs_lo", -c), const("obs_hi", c)], ["h0"]),
  ]
  n = int(meta["num_layers"])
  h = "h0"
  for i in range(n):
    w, b = const(f"layer{i}.w", arrays[f"layer{i}.w"]), const(f"layer{i}.b", arrays[f"layer{i}.b"])
    nodes += [helper.make_node("MatMul", [h, w], [f"mm{i}"]),
              helper.make_node("Add", [f"mm{i}", b], [f"z{i}"])]
    h = f"z{i}"
    if i < n - 1:
      if act_op:
        nodes.append(helper.make_node(act_op, [h], [f"a{i}"]))
      else:  # silu = x * sigmoid(x)
        nodes += [helper.make_node("Sigmoid", [h], [f"s{i}"]),
                  helper.make_node("Mul", [h, f"s{i}"], [f"a{i}"])]
      h = f"a{i}"
  nodes += [
      helper.make_node("Clip", [h, const("act_lo", -1.0), const("act_hi", 1.0)], ["action"]),
      helper.make_node("Mul", ["action", const("action_scale", meta["action_scale"])], ["delta"]),
      helper.make_node("Add", ["delta", const("home_ctrl", arrays["home_ctrl"])], ["raw_targets"]),
      helper.make_node("Max", ["raw_targets", const("ctrl_lo", arrays["ctrl_lo"])], ["targets_lo"]),
      helper.make_node("Min", ["targets_lo", const("ctrl_hi", arrays["ctrl_hi"])], ["joint_targets"]),
  ]
  obs_size, act_size = int(meta["obs_size"]), int(meta["action_size"])
  graph = helper.make_graph(
      nodes, "g1_rl_policy",
      [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", obs_size])],
      [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", act_size]),
       helper.make_tensor_value_info("joint_targets", TensorProto.FLOAT, ["batch", act_size])],
      inits)
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                            producer_name="g1_rl")
  model.ir_version = min(model.ir_version, 9)  # loadable by onnxruntime >= 1.16
  full_meta = dict(meta)
  full_meta.update({k: np.asarray(arrays[k], np.float64).tolist() for k in ROBOT_ARRAYS})
  helper.set_model_props(model, {META_KEY: json.dumps(full_meta)})
  onnx.checker.check_model(model)
  tmp = path + ".tmp"
  onnx.save(model, tmp)
  os.replace(tmp, path)
  return path


class OnnxPolicy:
  """Same interface as g1_rl.policy.Policy, backed by onnxruntime (no JAX needed)."""

  def __init__(self, path: str, providers=None):
    import onnxruntime as ort  # noqa: PLC0415

    avail = ort.get_available_providers()
    if providers is None:
      providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    self.session = ort.InferenceSession(path, providers=providers)
    self.provider = self.session.get_providers()[0]
    props = self.session.get_modelmeta().custom_metadata_map
    if META_KEY not in props:
      raise ValueError(f"{path} has no '{META_KEY}' metadata (not exported by g1_rl)")
    self.meta = json.loads(props[META_KEY])
    self.home = np.asarray(self.meta["home_ctrl"])
    self.lo, self.hi = np.asarray(self.meta["ctrl_lo"]), np.asarray(self.meta["ctrl_hi"])
    self.kp, self.kd = np.asarray(self.meta["kp"]), np.asarray(self.meta["kd"])
    self.action_scale = np.asarray(self.meta["action_scale"], np.float64)

  def _run(self, obs):
    x = np.asarray(obs, np.float32)
    single = x.ndim == 1
    action, targets = self.session.run(None, {"obs": x[None] if single else x})
    return (action[0], targets[0]) if single else (action, targets)

  def __call__(self, obs):
    return self._run(obs)[0]

  def joint_targets(self, obs):
    return self._run(obs)[1]


def load_any(path: str, device=None):
  """Policy for a .npz (JAX) or .onnx (onnxruntime) file."""
  if path.endswith(".onnx"):
    return OnnxPolicy(path)
  from g1_rl.policy import Policy  # noqa: PLC0415
  return Policy.load(path, device=device)


# ----------------------------------------------------------------------------- bundle
def verify(arrays, meta, onnx_path, n: int = 4096, seed: int = 0, jax_policy=None) -> dict:
  """Max |difference| between NumPy, ONNX and (optionally) the JAX policy on random
  observations spread like the training data (plus far-out ones to exercise the clips)."""
  rng = np.random.default_rng(seed)
  std = np.sqrt(arrays["obs_var"] + 1e-8)
  obs = arrays["obs_mean"] + std * rng.standard_normal((n, len(std))) * rng.choice([0.5, 1, 3, 20], (n, 1))
  obs = obs.astype(np.float32)
  ref = numpy_forward(arrays, meta, obs)
  pi = OnnxPolicy(onnx_path, providers=["CPUExecutionProvider"])
  a_onnx, t_onnx = pi._run(obs)
  t_ref = np.clip(arrays["home_ctrl"] + np.asarray(meta["action_scale"]) * ref, arrays["ctrl_lo"], arrays["ctrl_hi"])
  out = {"samples": n, "onnx_vs_numpy_action": float(np.abs(a_onnx - ref).max()),
         "onnx_vs_numpy_targets": float(np.abs(t_onnx - t_ref).max()),
         "saturated_fraction": float((np.abs(ref) >= 1).mean())}
  if jax_policy is not None:
    out["jax_vs_numpy_action"] = float(np.abs(np.asarray(jax_policy(obs)) - ref).max())
  return out


DEPLOY_README = """# G1 walking policy: deployment bundle

Exported by g1_rl {version} from `{source}` (iteration {iteration}, {steps:.0f}M training steps).

| file | what |
|---|---|
| `policy.onnx` | the controller; robot constants embedded (metadata key `g1_rl`) |
| `policy.npz` + `policy.json` | the same policy in the g1_rl format (`g1_rl.policy.Policy`) |
| `run_mujoco.py` | standalone runner: needs only `numpy mujoco onnxruntime` |
| `export_info.json` | metadata and the numerical check of the ONNX graph |

## Run it

```bash
pip install numpy mujoco onnxruntime          # or onnxruntime-gpu for CUDA
python run_mujoco.py --xml <menagerie>/unitree_g1/scene.xml --viewer      # keyboard control
python run_mujoco.py --xml <menagerie>/unitree_g1/scene.xml --cmd 0.5 0 0 --seconds 10
```

## Interface (50 Hz)

Input `obs` (batch, {obs_size}) float32, in this order:
{layout}

* gyro: base angular velocity in the body frame (rad/s); gravity: unit gravity vector in the body frame
* command: (vx, vy, wz) in m/s, m/s, rad/s; trained ranges vx [-0.6, 1.0], vy [-0.5, 0.5], wz [-1, 1]
* joint_pos is relative to the home pose; joint order is `joint_names` in the metadata
* last_act: the previous `action` output
* gait: cos, sin of two foot phases (left starts at 0, right at pi), advancing by
  2*pi*f*dt per step, f = {gait_freq} Hz

Outputs: `action` in [-1, 1] and `joint_targets` (rad) = clip(home + action_scale * action, ctrl_lo, ctrl_hi),
action_scale = {scale},
tracked by a PD loop with kp, kd from the metadata (torque = kp*(target - q) - kd*qdot).
"""


def _scale_text(meta) -> str:
  s = np.atleast_1d(np.asarray(meta["action_scale"], np.float64))
  if np.all(s == s[0]):
    return f"{s[0]:g} rad for every joint"
  groups = {}
  for n, v in zip(meta["joint_names"], np.broadcast_to(s, (len(meta["joint_names"]),))):
    groups.setdefault(f"{v:g}", []).append(n.replace("_joint", ""))
  return "; ".join(f"{v} rad: {', '.join(ns)}" for v, ns in groups.items())


def export_bundle(policy_npz: str, out_dir: str, include_model: str | None = None,
                  jax_policy=None) -> dict:
  """Write the deployment bundle and return export_info."""
  import g1_rl  # noqa: PLC0415

  arrays, meta = load_npz(policy_npz)
  os.makedirs(out_dir, exist_ok=True)
  onnx_path = to_onnx(arrays, meta, os.path.join(out_dir, "policy.onnx"))
  shutil.copy(policy_npz, os.path.join(out_dir, "policy.npz"))
  shutil.copy(policy_npz[:-4] + ".json", os.path.join(out_dir, "policy.json"))
  runner_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "deploy", "run_mujoco.py")
  shutil.copy(runner_src, os.path.join(out_dir, "run_mujoco.py"))
  if include_model:
    shutil.copytree(include_model, os.path.join(out_dir, "unitree_g1"), dirs_exist_ok=True)
  check = verify(arrays, meta, onnx_path, jax_policy=jax_policy)
  info = {"g1_rl_version": g1_rl.__version__, "source": os.path.abspath(policy_npz),
          "iteration": meta.get("iteration"), "env_steps": meta.get("env_steps"),
          "onnx_bytes": os.path.getsize(onnx_path), "check": check}
  with open(os.path.join(out_dir, "export_info.json"), "w") as f:
    json.dump(info, f, indent=2)
  layout = "\n".join(f"  {k:<10} [{a}:{b}]" for k, (a, b) in meta["obs_layout"].items())
  with open(os.path.join(out_dir, "README.md"), "w") as f:
    f.write(DEPLOY_README.format(
        version=g1_rl.__version__, source=os.path.basename(policy_npz), iteration=meta.get("iteration"),
        steps=(meta.get("env_steps") or 0) / 1e6, obs_size=meta["obs_size"], layout=layout,
        gait_freq=float(np.mean(meta.get("gait_freq_range", [1.25, 1.5]))), scale=_scale_text(meta)))
  return info
