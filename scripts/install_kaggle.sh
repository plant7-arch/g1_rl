#!/usr/bin/env bash
# Install g1_rl + GPU JAX + MuJoCo/MJX on a Kaggle "GPU T4 x2" notebook
# (Internet must be ON in the notebook settings).
#
# Override versions if a release breaks something, e.g.
#   JAX_SPEC='jax[cuda12]==0.11.2' MJ_SPEC='mujoco-mjx==3.16.0' bash scripts/install_kaggle.sh
# (those are the versions this package was developed and tested against).
set -euo pipefail
cd "$(dirname "$0")/.."

JAX_SPEC="${JAX_SPEC:-jax[cuda12]}"
MJ_SPEC="${MJ_SPEC:-mujoco-mjx}"

echo ">> Installing ${JAX_SPEC} ${MJ_SPEC} ..."
# Kaggle ships older jax/torch/tf; we only need jax here. Pip may warn about
# torch/tensorflow CUDA pins: harmless, this pipeline never imports them.
pip install -q -U "${JAX_SPEC}" "${MJ_SPEC}" mediapy 2>&1 | grep -v -i "dependency conflicts\|requires\|incompatible" || true
pip install -q -e . --no-deps

python - <<'EOF'
import jax, mujoco
gpus = [d for d in jax.devices() if d.platform == "gpu"]
print(f"jax {jax.__version__}  mujoco {mujoco.__version__}  GPUs: {[d.device_kind for d in gpus]}")
if not gpus:
    raise SystemExit("!! JAX sees no GPU. Check Accelerator = 'GPU T4 x2' and re-run.")
EOF
