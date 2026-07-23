"""Test suite for the C++ Generator module."""

from ml_switcheroo.core.compiler.backends.cpp.generator import TorchCppExtensionGenerator


def test_torch_extension_generator():
  """Verifies Torch C++ Extension Generator."""
  gen = TorchCppExtensionGenerator(module_name="my_custom_op")

  args = [{"name": "input", "type": "torch::Tensor"}, {"name": "weights", "type": "torch::Tensor"}]
  body = ["auto result = input * weights;", "return result;"]

  fwd = gen.generate_forward_function(args, body)
  assert fwd.name == "forward"
  assert len(fwd.arguments) == 2
  assert fwd.arguments[0].name == "input"

  mod = gen.build_module(fwd)
  text = mod.to_text()

  assert "#include <torch/extension.h>" in text
  assert "torch::Tensor forward(torch::Tensor input, torch::Tensor weights)" in text
  assert "auto result = input * weights;" in text
  assert "PYBIND11_MODULE" in text
  assert 'm.def("forward"' in text
