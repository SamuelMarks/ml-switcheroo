"""Test suite for the Torch Io module."""

from ml_switcheroo.frameworks.torch_io import TorchIOMixin


def test_torch_io_methods():
  """Verifies the behavior of all PyTorch I/O mixin methods."""
  mixin = TorchIOMixin()

  # Test get_serialization_imports
  assert mixin.get_serialization_imports() == ["import torch"]

  # Test get_serialization_syntax
  assert mixin.get_serialization_syntax("save", "my_file.pth", "my_model") == "torch.save(my_model, my_file.pth)"
  assert mixin.get_serialization_syntax("load", "my_file.pth") == "torch.load(my_file.pth)"
  assert mixin.get_serialization_syntax("invalid", "my_file.pth") == ""

  # Test get_weight_conversion_imports
  assert mixin.get_weight_conversion_imports() == ["import torch"]

  # Test get_weight_load_code
  load_code = mixin.get_weight_load_code("path_to_checkpoint")
  assert "torch.load(path_to_checkpoint, map_location='cpu')" in load_code
  assert "loaded['state_dict']" in load_code

  # Test get_tensor_to_numpy_expr
  assert mixin.get_tensor_to_numpy_expr("my_tensor") == "my_tensor.detach().cpu().numpy()"

  # Test get_weight_save_code
  save_code = mixin.get_weight_save_code("my_state", "out_path")
  assert "torch.from_numpy" in save_code
  assert "torch.save(converted_state, out_path)" in save_code
