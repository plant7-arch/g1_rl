"""Device selection: GPU by default, CPU as an explicit or automatic fallback.

Call ``setup()`` BEFORE anything else imports jax (scripts do this on line 1),
because XLA flags and memory settings are read when the backend initialises.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import warnings


@dataclasses.dataclass
class Runtime:
  platform: str  # "gpu" | "cpu"
  devices: list
  mesh: object  # jax.sharding.Mesh with a single axis named "batch"

  @property
  def n_devices(self) -> int:
    return len(self.devices)

  def describe(self) -> str:
    names = ", ".join(getattr(d, "device_kind", str(d)) for d in self.devices)
    return f"{self.platform.upper()} x{self.n_devices} [{names}]"


def _set_env_defaults(platform: str) -> None:
  if platform in ("auto", "gpu"):
    flags = os.environ.get("XLA_FLAGS", "")
    # Recommended by the MJX docs for better GPU kernels.
    if "--xla_gpu_triton_gemm_any" not in flags:
      flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = flags.strip()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
  # MuJoCo offscreen rendering without a display (Kaggle / headless servers).
  os.environ.setdefault("MUJOCO_GL", "egl")


def setup(platform: str = "auto", max_devices: int | None = None,
          compile_cache: str | None = None) -> Runtime:
  """Initialise JAX on the requested platform.

  platform: "auto" (GPU if available, else CPU with a warning), "gpu" (fail if
  no GPU), or "cpu".
  max_devices: use only the first N devices (e.g. 1 to benchmark a single T4).
  compile_cache: directory for JAX's persistent compilation cache, so a resumed
  Kaggle session skips most of the multi-minute MJX compile.
  """
  platform = platform.lower()
  if platform not in ("auto", "gpu", "cpu"):
    raise ValueError(f"platform must be auto|gpu|cpu, got {platform!r}")
  if "jax" in sys.modules and platform == "cpu" and os.environ.get("JAX_PLATFORMS") != "cpu":
    warnings.warn("jax was imported before device.setup(); forcing CPU may not take effect.")
  if platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
  _set_env_defaults(platform)

  import jax  # noqa: PLC0415  (import after env vars are set)

  # On Kaggle, /kaggle/working persists between sessions (Persistence: Files),
  # so the compiled-kernel cache survives and resumed runs skip most compiles.
  default_cache = ("/kaggle/working/.jax_cache" if os.path.isdir("/kaggle/working")
                   else os.path.expanduser("~/.cache/g1_rl/jax"))
  cache = compile_cache or os.environ.get("G1RL_COMPILE_CACHE", default_cache)
  try:
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
  except Exception as e:  # cache is an optimisation only
    warnings.warn(f"Compilation cache disabled: {e}")

  devices = []
  if platform in ("auto", "gpu"):
    try:
      devices = jax.devices("gpu")
    except RuntimeError:
      devices = []
    if not devices:
      if platform == "gpu":
        raise RuntimeError(
            "No GPU visible to JAX. Install jax[cuda12] (see scripts/install_kaggle.sh) "
            "or run with --platform cpu.")
      warnings.warn("No GPU found -> falling back to CPU (this will be slow).")
  if not devices:
    devices = jax.devices("cpu")
    platform = "cpu"
  else:
    platform = "gpu"
  if max_devices:
    devices = devices[:max_devices]

  import numpy as np  # noqa: PLC0415
  mesh = jax.sharding.Mesh(np.array(devices), ("batch",))
  return Runtime(platform=platform, devices=list(devices), mesh=mesh)
