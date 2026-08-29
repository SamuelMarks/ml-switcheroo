"""Unit tests for the C++ compiler backend parser (CppParser).

This module contains test cases to verify the correctness of the C++ parser implementation,
ensuring it correctly parses and extracts include directives, preprocessor macros,
function definitions, and pybind11 module definitions from C++ source code.
"""

from ml_switcheroo.core.compiler.backends.cpp.cst import CppModule, FunctionDefinition, MacroDefinition, PyBindModule
from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser


def test_parser_basic() -> None:
  """Tests basic parsing functionality of the CppParser.

  This test provides a block of C++ source code containing includes, a macro definition,
  a torch-extension style forward function declaration, and a pybind11 module block,
  then asserts that the CppParser extracts all of these constructs accurately with
  correct types, names, and arguments.

  Args:
      None

  Returns:
      None
  """
  code: str = """
#include <torch/extension.h>
#include "my_header.h"

#define MAX_VAL 100

torch::Tensor forward(torch::Tensor input, torch::Tensor weights) {
    auto result = input * weights;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "forward doc");
}
"""
  parser = CppParser(code)
  mod: CppModule = parser.parse()
  assert len(mod.includes) == 2
  assert mod.includes[0].path == "torch/extension.h"
  assert mod.includes[0].system is True
  assert mod.includes[1].path == "my_header.h"
  assert mod.includes[1].system is False

  macro_node = mod.body[0]
  assert isinstance(macro_node, MacroDefinition)
  assert macro_node.name == "MAX_VAL"
  assert macro_node.value == "100"

  func_node = mod.body[1]
  assert isinstance(func_node, FunctionDefinition)
  assert func_node.name == "forward"
  assert getattr(func_node.return_type, "name", None) == "torch::Tensor"
  assert len(func_node.arguments) == 2
  assert getattr(func_node.arguments[0].type_id, "name", None) == "torch::Tensor"

  pyb_node = mod.body[2]
  assert isinstance(pyb_node, PyBindModule)
  assert pyb_node.name == "TORCH_EXTENSION_NAME"
  assert pyb_node.module_var == "m"
  assert len(pyb_node.defs) == 1
  assert pyb_node.defs[0].name == "forward"
