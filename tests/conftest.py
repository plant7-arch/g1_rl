import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_rl import device  # noqa: E402

# G1RL_TEST_PLATFORM=gpu|cpu|auto (default auto: GPU if present, else CPU).
RUNTIME = device.setup(os.environ.get("G1RL_TEST_PLATFORM", "auto"))


def _emulate_donation_on_cpu():
  """XLA:CPU ignores buffer donation, so code that reads a donated input works on
  CPU but crashes on GPU ("Array has been deleted"). On CPU, make every
  parallel.jit_batched function delete its donated inputs after the call, so the
  test suite catches those bugs without a GPU."""
  import jax  # noqa: PLC0415

  from g1_rl import parallel  # noqa: PLC0415

  orig = parallel.jit_batched

  class _Deleting:
    def __init__(self, fn, donate):
      self._fn, self._donate = fn, tuple(donate)

    def __call__(self, *args):
      out = self._fn(*args)
      jax.block_until_ready(out)
      for i in self._donate:
        for leaf in jax.tree.leaves(args[i]):
          if isinstance(leaf, jax.Array) and not leaf.is_deleted():
            leaf.delete()
      return out

    def __getattr__(self, name):  # lower(), _cache_size(), ...
      return getattr(self._fn, name)

  def jit_batched(fn, mesh, donate_argnums=(), static_argnums=()):
    jitted = orig(fn, mesh, donate_argnums=donate_argnums, static_argnums=static_argnums)
    return _Deleting(jitted, donate_argnums) if donate_argnums else jitted

  parallel.jit_batched = jit_batched


if RUNTIME.platform == "cpu":
  _emulate_donation_on_cpu()
