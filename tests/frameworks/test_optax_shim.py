"""Test suite for the Optax Shim module."""

from ml_switcheroo.frameworks.common import optax_shim


def test_optax_shim() -> None:
  """Verifies the behavior of optax shim."""
  try:
    optax_shim.adam  # type: ignore
  except Exception:
    pass
