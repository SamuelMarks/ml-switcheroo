"""Test suite for the C++ CST module."""

import pytest
from ml_switcheroo.core.compiler.backends.cpp.cst import (
  CppNode,
  TypeIdentifier,
  VariableDeclaration,
  FunctionArgument,
  FunctionDefinition,
  RawStatement,
  IncludeDirective,
  CppModule,
)


def test_cpp_node_not_implemented():
  """Verifies that the base node raises NotImplementedError."""
  node = CppNode()
  with pytest.raises(NotImplementedError):
    node.to_text()


def test_type_identifier():
  """Verifies TypeIdentifier renders correctly."""
  t = TypeIdentifier(name="int")
  assert t.to_text() == "int"


def test_variable_declaration():
  """Verifies VariableDeclaration renders correctly."""
  t = TypeIdentifier(name="double")
  v1 = VariableDeclaration(type_id=t, name="x")
  assert v1.to_text() == "double x;"

  v2 = VariableDeclaration(type_id=t, name="y", initializer="1.0")
  assert v2.to_text() == "double y = 1.0;"


def test_function_argument():
  """Verifies FunctionArgument renders correctly."""
  t = TypeIdentifier(name="torch::Tensor")
  a = FunctionArgument(type_id=t, name="input")
  assert a.to_text() == "torch::Tensor input"


def test_function_definition():
  """Verifies FunctionDefinition renders correctly."""
  ret = TypeIdentifier(name="void")
  arg = FunctionArgument(type_id=TypeIdentifier(name="int"), name="x")
  body = [RawStatement(code="return;")]
  func = FunctionDefinition(return_type=ret, name="my_func", arguments=[arg], body=body)

  expected = "void my_func(int x) {\n" "    return;\n" "}"
  assert func.to_text() == expected


def test_include_directive():
  """Verifies IncludeDirective renders correctly."""
  inc1 = IncludeDirective(path="torch/extension.h", system=True)
  assert inc1.to_text() == "#include <torch/extension.h>"

  inc2 = IncludeDirective(path="local.h", system=False)
  assert inc2.to_text() == '#include "local.h"'


def test_cpp_module():
  """Verifies CppModule renders correctly."""
  inc = IncludeDirective(path="iostream", system=True)
  stmt = RawStatement(code="int x = 0;")
  mod = CppModule(includes=[inc], body=[stmt])

  expected = "#include <iostream>\n" "\n" "int x = 0;\n"
  assert mod.to_text() == expected
