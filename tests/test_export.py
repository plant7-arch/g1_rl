"""Phase 6: ONNX export matches the training policy, numerically and in closed loop.

Uses a randomly initialised policy (no training needed); runs on CPU in seconds.
"""

import json
import os
import subprocess
import sys

import jax
import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from conftest import RUNTIME  # noqa: E402

from g1_rl import checkpoint, export, ppo  # noqa: E402
from g1_rl.config import PPOConfig, SimConfig  # noqa: E402
from g1_rl.env import G1Env  # noqa: E402
from g1_rl.policy import Policy  # noqa: E402
from g1_rl.sim2sim import MujocoRunner  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
  env, cfg = G1Env(), PPOConfig()
  ts = ppo.init_train_state(jax.random.PRNGKey(3), env.obs_size, env.priv_obs_size, env.action_size, cfg)
  # Non-trivial normaliser and a larger last layer, so the clips actually matter.
  rng = np.random.default_rng(0)
  stats = ts.obs_stats._replace(mean=rng.normal(size=env.obs_size).astype(np.float32),
                                var=rng.uniform(0.05, 3.0, env.obs_size).astype(np.float32))
  net = [dict(l) for l in ts.params["policy"]["net"]]
  net[-1]["w"] = net[-1]["w"] * 300.0
  ts = ts._replace(obs_stats=stats, params={**ts.params, "policy": {**ts.params["policy"], "net": net}})
  d = tmp_path_factory.mktemp("export")
  npz = checkpoint.export_policy(str(d / "policy_final.npz"), ts, env, cfg, {"env_steps": 0})
  pi = Policy.load(npz, device=RUNTIME.devices[0])
  info = export.export_bundle(npz, str(d / "bundle"), jax_policy=pi)
  return d / "bundle", npz, pi, info


def test_onnx_matches_reference(bundle):
  out, _, _, info = bundle
  c = info["check"]
  assert c["onnx_vs_numpy_action"] < 1e-4 and c["jax_vs_numpy_action"] < 1e-4
  assert c["onnx_vs_numpy_targets"] < 1e-4
  assert 0.01 < c["saturated_fraction"] < 0.99  # the output clip was exercised
  assert sorted(os.listdir(out)) == ["README.md", "export_info.json", "policy.json", "policy.npz",
                                     "policy.onnx", "run_mujoco.py"]


def test_onnx_file_is_self_contained(bundle):
  out, npz, pi, _ = bundle
  op = export.OnnxPolicy(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
  meta = json.load(open(npz[:-4] + ".json"))
  for k in ("obs_layout", "joint_names", "action_scale", "ctrl_dt", "obs_size"):
    assert op.meta[k] == meta[k]
  np.testing.assert_allclose(op.home, pi.home, atol=1e-6)
  np.testing.assert_allclose(op.kp, pi.kp, atol=1e-6)
  np.testing.assert_allclose(op.kd, pi.kd, atol=1e-6)
  obs = np.random.default_rng(1).normal(size=(5, meta["obs_size"])).astype(np.float32)
  np.testing.assert_allclose(op(obs), np.asarray(pi(obs)), atol=1e-5)
  np.testing.assert_allclose(op(obs[0]), np.asarray(pi(obs[0])), atol=1e-5)  # unbatched
  np.testing.assert_allclose(op.joint_targets(obs), pi.joint_targets(obs), atol=1e-5)


def test_same_closed_loop_in_mujoco(bundle):
  """The ONNX and the JAX policy produce the same trajectory in the sim-to-sim runner."""
  out, _, pi, _ = bundle
  op = export.OnnxPolicy(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
  xml = SimConfig().xml_path
  runs = [MujocoRunner(xml, p) for p in (pi, op)]
  for r in runs:
    r.reset(command=(0.3, 0.0, 0.0))
  for _ in range(5):
    acts = [r.step() for r in runs]
    np.testing.assert_allclose(acts[0], acts[1], atol=1e-4)
  np.testing.assert_allclose(runs[0].d.qpos, runs[1].d.qpos, atol=1e-4)


def test_standalone_runner(bundle):
  """deploy/run_mujoco.py runs from the bundle without g1_rl on the path."""
  out, _, _, _ = bundle
  env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
  r = subprocess.run([sys.executable, "-I", str(out / "run_mujoco.py"), "--xml", SimConfig().xml_path,
                      "--cmd", "0", "0", "0", "--seconds", "0.2"],
                     capture_output=True, text=True, cwd=str(out), env=env, timeout=300)
  assert "policy: iteration" in r.stdout, r.stderr[-2000:]
  # A random policy may fall; the point is that it runs and reports.
  assert ("walked" in r.stdout) or ("fell" in r.stdout), r.stdout + r.stderr[-2000:]
