"""Test module."""

from ml_switcheroo.core.compiler.backends.cpp.generator import TorchCppExtensionGenerator
from ml_switcheroo.core.compiler.backends.cpp.cst import FunctionDefinition, ReturnStatement


def test_generator_init():
  """Test element."""
  gen = TorchCppExtensionGenerator(module_name="my_module")
  assert gen.module_name == "my_module"
  assert len(gen.includes) == 2
  assert gen.includes[0].path == "torch/extension.h"
  assert gen.includes[1].path == "vector"


def test_generator_forward_function():
  """Test element."""
  gen = TorchCppExtensionGenerator(module_name="my_module")
  args = [{"name": "x", "type": "torch::Tensor"}]
  body = [ReturnStatement()]
  func = gen.generate_forward_function(args, body)

  assert isinstance(func, FunctionDefinition)
  assert func.name == "forward"
  assert func.return_type.name == "torch::Tensor"
  assert len(func.arguments) == 1
  assert func.arguments[0].name == "x"
  assert func.arguments[0].type_id.name == "torch::Tensor"
  assert len(func.body) == 1


def test_generator_pybind_module():
  """Test element."""
  gen = TorchCppExtensionGenerator(module_name="my_module")
  pybind = gen.generate_pybind_module("forward")
  assert pybind.name == "TORCH_EXTENSION_NAME"
  assert pybind.module_var == "m"
  assert len(pybind.defs) == 1
  assert pybind.defs[0].name == "forward"
  assert pybind.defs[0].function_ref == "forward"
  assert pybind.defs[0].docstring == "my_module forward"


def test_generator_build_module():
  """Test element."""
  gen = TorchCppExtensionGenerator(module_name="my_module")
  args = [{"name": "x", "type": "torch::Tensor"}]
  body = [ReturnStatement()]
  func = gen.generate_forward_function(args, body)

  mod = gen.build_module(func)
  assert len(mod.includes) == 2
  assert len(mod.body) == 2
  assert mod.body[0] == func
  assert mod.body[1].name == "TORCH_EXTENSION_NAME"
