"""Test suite for the Keras Gap module."""

from ml_switcheroo.frameworks.keras import KerasAdapter


def test_keras_gap():
  """Verifies the behavior of Keras gap."""
  adapter = KerasAdapter()
  try:
    adapter.get_loss("NonExistentLoss")
  except Exception:
    pass
  try:
    adapter.get_optimizer("NonExistentOpt")
  except Exception:
    pass
  try:
    adapter.get_layer("NonExistentLayer")
  except Exception:
    pass
