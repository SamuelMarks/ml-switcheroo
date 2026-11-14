"""Unit tests for `TorchCppExtensionGenerator` (`ml_switcheroo.core.compiler.backends.cpp.generator`)."""

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  BinaryExpression,
  FunctionDefinition,
  Identifier,
  PyBindModule,
  ReturnStatement,
  TypeIdentifier,
  VariableDeclaration,
)
from ml_switcheroo.core.compiler.backends.cpp.generator import TorchCppExtensionGenerator


def test_torch_cpp_extension_generator_init_forward_and_module() -> None:
  """Verify `TorchCppExtensionGenerator` produces valid forward functions, pybind11 blocks, and C++ modules."""
  gen = TorchCppExtensionGenerator(module_name="my_custom_op")
  assert gen.module_name == "my_custom_op"
  assert [inc.path for inc in gen.includes] == ["torch/extension.h", "vector"]

  args = [{"name": "input", "type": "torch::Tensor"}, {"name": "weights", "type": "torch::Tensor"}]
  body = [
    VariableDeclaration(
      type_id=TypeIdentifier("auto"),
      name="result",
      initializer=BinaryExpression(left=Identifier("input"), operator="*", right=Identifier("weights")),
    ),
    ReturnStatement(value=Identifier("result")),
  ]

  fwd = gen.generate_forward_function(args, body)
  assert isinstance(fwd, FunctionDefinition)
  assert fwd.name == "forward"
  assert fwd.return_type.name == "torch::Tensor"
  assert [(a.name, a.type_id.name) for a in fwd.arguments] == [
    ("input", "torch::Tensor"),
    ("weights", "torch::Tensor"),
  ]

  pybind = gen.generate_pybind_module("forward")
  assert pybind.name == "TORCH_EXTENSION_NAME"
  assert pybind.module_var == "m"
  assert pybind.defs[0].docstring == "my_custom_op forward"

  mod = gen.build_module(fwd)
  assert len(mod.includes) == 2
  assert mod.body[0] is fwd
  pb_node = mod.body[1]
  assert isinstance(pb_node, PyBindModule)
  assert pb_node.name == "TORCH_EXTENSION_NAME"

  text = mod.to_text()
  assert "#include <torch/extension.h>" in text
  assert "torch::Tensor forward(torch::Tensor input, torch::Tensor weights)" in text
  assert "auto result = input * weights;" in text
  assert 'm.def("forward"' in text
