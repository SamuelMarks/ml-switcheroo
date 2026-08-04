"""Test module."""

from ml_switcheroo.frameworks.keras_io import KerasIOMixin


def test_keras_io_mixin_serialization():
  """Test function."""
  mixin = KerasIOMixin()

  assert mixin.get_serialization_imports() == ["import keras"]
  assert mixin.get_serialization_syntax("save", "file", "model") == "model.save(file)"
  assert mixin.get_serialization_syntax("load", "file") == "keras.saving.load_model(file)"
  assert mixin.get_serialization_syntax("invalid", "file") == ""
  assert mixin.get_serialization_syntax("save", "file", None) == ""

  assert mixin.get_weight_conversion_imports() == ["import keras", "import numpy as np", "import h5py"]

  load_code = mixin.get_weight_load_code("path")
  assert "model = keras.models.load_model(path, compile=False)" in load_code
  assert "import h5py" in load_code

  expr = mixin.get_tensor_to_numpy_expr("t")
  assert expr == "t.numpy() if hasattr(t, 'numpy') else np.array(t)"

  save_code = mixin.get_weight_save_code("state", "path")
  assert 'with h5py.File(path, "w") as f:' in save_code
