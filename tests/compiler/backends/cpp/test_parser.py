"""Unit tests for the C++ compiler backend parser (CppParser).

This module contains test cases to verify the correctness of the C++ parser implementation,
ensuring it correctly parses and extracts include directives, preprocessor macros,
function definitions, and pybind11 module definitions from C++ source code.
"""

from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser


def test_parser_basic():
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
  code = """
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
  mod = parser.parse()
  assert len(mod.includes) == 2
  assert mod.includes[0].path == "torch/extension.h"
  assert mod.includes[0].system is True
  assert mod.includes[1].path == "my_header.h"
  assert mod.includes[1].system is False

  assert mod.body[0].name == "MAX_VAL"
  assert mod.body[0].value == "100"

  assert mod.body[1].name == "forward"
  assert mod.body[1].return_type.name == "torch::Tensor"
  assert len(mod.body[1].arguments) == 2
  assert mod.body[1].arguments[0].type_id.name == "torch::Tensor"

  assert mod.body[2].name == "TORCH_EXTENSION_NAME"
  assert mod.body[2].module_var == "m"
  assert len(mod.body[2].defs) == 1
  assert mod.body[2].defs[0].name == "forward"
