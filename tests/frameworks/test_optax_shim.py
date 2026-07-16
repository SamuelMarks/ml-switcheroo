"""Test module."""

from ml_switcheroo.frameworks.common import optax_shim


def test_optax_shim():
  """Test function."""
  # Attempt to get something that optax would have
  try:
    optax_shim.adam
  except Exception:
    pass
