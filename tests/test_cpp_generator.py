"""Test module."""

from ml_switcheroo.core.compiler.backends.cpp.generator import TorchCppExtensionGenerator
from ml_switcheroo.core.compiler.backends.cpp.cst import FunctionDefinition, ReturnStatement, PyBindModule, CppModule
from typing import List, Dict


def test_generator_init() -> None:
  """Test element."""
  gen: TorchCppExtensionGenerator = TorchCppExtensionGenerator(module_name="my_module")
  assert gen.module_name == "my_module"
  assert len(gen.includes) == 2
  assert gen.includes[0].path == "torch/extension.h"
  assert gen.includes[1].path == "vector"


def test_generator_forward_function() -> None:
  """Test element."""
  gen: TorchCppExtensionGenerator = TorchCppExtensionGenerator(module_name="my_module")
  args: List[Dict[str, str]] = [{"name": "x", "type": "torch::Tensor"}]
  body: List[ReturnStatement] = [ReturnStatement()]
  func: FunctionDefinition = gen.generate_forward_function(args, body)

  assert isinstance(func, FunctionDefinition)
  assert func.name == "forward"
  assert func.return_type.name == "torch::Tensor"
  assert len(func.arguments) == 1
  assert func.arguments[0].name == "x"
  assert func.arguments[0].type_id.name == "torch::Tensor"
  assert len(func.body) == 1


def test_generator_pybind_module() -> None:
  """Test element."""
  gen: TorchCppExtensionGenerator = TorchCppExtensionGenerator(module_name="my_module")
  pybind: PyBindModule = gen.generate_pybind_module("forward")
  assert pybind.name == "TORCH_EXTENSION_NAME"
  assert pybind.module_var == "m"
  assert len(pybind.defs) == 1
  assert pybind.defs[0].name == "forward"
  assert pybind.defs[0].function_ref == "forward"
  assert pybind.defs[0].docstring == "my_module forward"


def test_generator_build_module() -> None:
  """Test element."""
  gen: TorchCppExtensionGenerator = TorchCppExtensionGenerator(module_name="my_module")
  args: List[Dict[str, str]] = [{"name": "x", "type": "torch::Tensor"}]
  body: List[ReturnStatement] = [ReturnStatement()]
  func: FunctionDefinition = gen.generate_forward_function(args, body)

  mod: CppModule = gen.build_module(func)
  assert len(mod.includes) == 2
  assert len(mod.body) == 2
  assert mod.body[0] == func
  assert mod.body[1].name == "TORCH_EXTENSION_NAME"
