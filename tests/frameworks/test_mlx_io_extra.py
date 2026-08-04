"""Test module."""

from ml_switcheroo.frameworks.mlx_io import MlxIOMixin


def test_mlx_io_mixin_serialization():
  """Test function."""
  mixin = MlxIOMixin()

  assert mixin.get_serialization_imports() == ["import mlx.core as mx"]
  assert mixin.get_serialization_syntax("save", "file", "model") == "mx.save(file, model)"
  assert mixin.get_serialization_syntax("load", "file") == "mx.load(file)"
  assert mixin.get_serialization_syntax("invalid", "file") == ""
  assert mixin.get_serialization_syntax("save", "file", None) == ""

  assert mixin.get_weight_conversion_imports() == ["import mlx.core as mx"]

  load_code = mixin.get_weight_load_code("path")
  assert "mx.load(path)" in load_code

  expr = mixin.get_tensor_to_numpy_expr("t")
  assert expr == "np.array(t)"

  save_code = mixin.get_weight_save_code("state", "path")
  assert "mlx_state =" in save_code
