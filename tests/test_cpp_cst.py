"""Test module."""

import pytest
from ml_switcheroo.core.compiler.backends.cpp.cst import (
  CppNode,
  TypeIdentifier,
  Identifier,
  BinaryExpression,
  MethodCall,
  ReturnStatement,
  VariableDeclaration,
  FunctionArgument,
  FunctionDefinition,
  RawStatement,
  MacroDefinition,
  BlockStatement,
  PyBindDef,
  PyBindModule,
  IncludeDirective,
  CppModule,
)


def test_cpp_node_base():
  """Test element."""
  node = CppNode()
  with pytest.raises(NotImplementedError):
    node.to_text()


def test_type_identifier():
  """Test element."""
  node = TypeIdentifier(name="int")
  assert node.to_text() == "int"


def test_identifier():
  """Test element."""
  node = Identifier(name="a")
  assert node.to_text() == "a"


def test_binary_expression():
  """Test element."""
  node = BinaryExpression(left=Identifier("a"), operator="+", right=Identifier("b"))
  assert node.to_text() == "a + b"


def test_method_call():
  """Test element."""
  node = MethodCall(name="foo")
  assert node.to_text() == "foo()"
  node_args = MethodCall(name="foo", arguments=[Identifier("a"), Identifier("b")])
  assert node_args.to_text() == "foo(a, b)"


def test_return_statement():
  """Test element."""
  node = ReturnStatement()
  assert node.to_text() == "return;"
  node_val = ReturnStatement(value=Identifier("a"))
  assert node_val.to_text() == "return a;"


def test_variable_declaration():
  """Test element."""
  node = VariableDeclaration(type_id=TypeIdentifier("int"), name="a")
  assert node.to_text() == "int a;"
  node_init = VariableDeclaration(type_id=TypeIdentifier("int"), name="a", initializer=Identifier("1"))
  assert node_init.to_text() == "int a = 1;"
  node_init_str = VariableDeclaration(type_id=TypeIdentifier("int"), name="a", initializer="1")
  assert node_init_str.to_text() == "int a = 1;"


def test_function_argument():
  """Test element."""
  node = FunctionArgument(type_id=TypeIdentifier("int"), name="a")
  assert node.to_text() == "int a"


def test_function_definition():
  """Test element."""
  node = FunctionDefinition(
    return_type=TypeIdentifier("int"),
    name="foo",
    arguments=[FunctionArgument(type_id=TypeIdentifier("int"), name="a")],
    body=[ReturnStatement(value=Identifier("a"))],
  )
  assert node.to_text() == "int foo(int a) {\n    return a;\n}"


def test_raw_statement():
  """Test element."""
  node = RawStatement(code="int a = 1;")
  assert node.to_text() == "int a = 1;"


def test_macro_definition():
  """Test element."""
  node = MacroDefinition(name="FOO", value="1")
  assert node.to_text() == "#define FOO 1"


def test_block_statement():
  """Test element."""
  node = BlockStatement(statements=[ReturnStatement()])
  assert node.to_text() == "{\n    return;\n}"


def test_pybind_def():
  """Test element."""
  node = PyBindDef(name="foo", function_ref="foo_impl", docstring="foo doc")
  assert node.to_text() == 'm.def("foo", &foo_impl, "foo doc");'


def test_pybind_module():
  """Test element."""
  node = PyBindModule(
    name="foo", module_var="m", defs=[PyBindDef(name="foo", function_ref="foo_impl", docstring="foo doc")]
  )
  assert node.to_text() == 'PYBIND11_MODULE(foo, m) {\n    m.def("foo", &foo_impl, "foo doc");\n}'


def test_include_directive():
  """Test element."""
  node = IncludeDirective(path="iostream", system=True)
  assert node.to_text() == "#include <iostream>"
  node_local = IncludeDirective(path="foo.h", system=False)
  assert node_local.to_text() == '#include "foo.h"'

  with pytest.raises(ValueError):
    IncludeDirective(path="")

  with pytest.raises(ValueError):
    IncludeDirective(path=123)

  with pytest.raises(ValueError):
    IncludeDirective(path="<iostream>")

  with pytest.raises(ValueError):
    IncludeDirective(path="foo.h> ")

  with pytest.raises(ValueError):
    IncludeDirective(path='"foo.h"')


def test_cpp_module():
  """Test element."""
  node = CppModule(includes=[IncludeDirective(path="iostream", system=True)], body=[ReturnStatement()])
  assert node.to_text() == "#include <iostream>\n\nreturn;\n"

  node_empty = CppModule()
  assert node_empty.to_text() == "\n"


def test_cpp_node_parse():
  """Test element."""
  # Will use the parser
  node = CppNode.parse("int a = 1;")
  assert isinstance(node, CppModule)
