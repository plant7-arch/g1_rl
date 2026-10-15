import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402

# G1RL_TEST_PLATFORM=gpu|cpu|auto (default auto: GPU if present, else CPU).
RUNTIME = device.setup(os.environ.get("G1RL_TEST_PLATFORM", "auto"))
