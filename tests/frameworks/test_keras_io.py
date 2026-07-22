"""Test suite for the Keras Io module."""

from ml_switcheroo.frameworks.keras_io import KerasIOMixin


def test_keras_io_dummy():
  """Verifies the behavior of Keras I/O dummy."""
  mixin = KerasIOMixin()
  assert hasattr(mixin, "get_serialization_imports")
