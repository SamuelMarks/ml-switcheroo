"""C++ Code Generation Utilities.

Builds specific C++ module structures (like PyTorch C++ extensions)
using the structural CST API.
"""

from typing import Dict, List
from ml_switcheroo.core.compiler.backends.cpp.cst import (
  CppModule,
  IncludeDirective,
  FunctionDefinition,
  FunctionArgument,
  TypeIdentifier,
  PyBindDef,
  PyBindModule,
  CppNode,
)


class TorchCppExtensionGenerator:
  """Generates a complete torch.nn.cpp.ModuleWrapper extension."""

  def __init__(self, module_name: str) -> None:
    """Initializes the generator.

    Args:
        module_name: The name of the Python module to be generated.
    """
    self.module_name = module_name
    self.includes = [
      IncludeDirective(path="torch/extension.h", system=True),
      IncludeDirective(path="vector", system=True),
    ]

  def generate_forward_function(self, args: List[Dict[str, str]], body_nodes: List[CppNode]) -> FunctionDefinition:
    """Generates the main forward pass function.

    Args:
        args: List of dictionaries with 'name' and 'type' keys.
        body_nodes: List of CppNode statements.

    Returns:
        FunctionDefinition: The constructed function.
    """
    cpp_args = [FunctionArgument(type_id=TypeIdentifier(name=arg["type"]), name=arg["name"]) for arg in args]

    # For torch wrappers, we typically return a tensor or vector of tensors
    ret_type = TypeIdentifier(name="torch::Tensor")

    return FunctionDefinition(return_type=ret_type, name="forward", arguments=cpp_args, body=body_nodes)

  def generate_pybind_module(self, func_name: str) -> PyBindModule:
    """Generates the PYBIND11_MODULE block binding the C++ function to Python.

    Args:
        func_name: The name of the C++ function to bind.

    Returns:
        PyBindModule: The pybind module definition node.
    """
    d = PyBindDef(name="forward", function_ref=func_name, docstring=f"{self.module_name} forward")
    return PyBindModule(name="TORCH_EXTENSION_NAME", module_var="m", defs=[d])

  def build_module(self, forward_func: FunctionDefinition) -> CppModule:
    """Assembles the full C++ module.

    Args:
        forward_func: The primary function to expose.

    Returns:
        CppModule: The root CST node.
    """
    pybind = self.generate_pybind_module(forward_func.name)

    return CppModule(includes=self.includes, body=[forward_func, pybind])
