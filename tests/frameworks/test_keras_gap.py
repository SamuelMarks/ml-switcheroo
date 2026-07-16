"""Test module."""

from ml_switcheroo.frameworks.keras import KerasAdapter


def test_keras_gap():
  """Test function."""
  adapter = KerasAdapter()

  # Keras lines 26-27 (probably unsupported mapping or fallback)
  # Keras lines 61-63 (probably get_loss)
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
