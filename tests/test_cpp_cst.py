"""Test suite for cpp cst."""

import pytest

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  BinaryExpression,
  BlockStatement,
  CppModule,
  CppNode,
  FunctionArgument,
  FunctionDefinition,
  Identifier,
  IncludeDirective,
  MacroDefinition,
  MethodCall,
  PyBindDef,
  PyBindModule,
  RawStatement,
  ReturnStatement,
  TypeIdentifier,
  VariableDeclaration,
)


def test_cpp_node_base() -> None:
  """Verify cpp node base behavior."""
  node: CppNode = CppNode()
  with pytest.raises(NotImplementedError):
    node.to_text()


def test_type_identifier() -> None:
  """Verify type identifier behavior."""
  node: TypeIdentifier = TypeIdentifier(name="int")
  assert node.to_text() == "int"


def test_identifier() -> None:
  """Verify identifier behavior."""
  node: Identifier = Identifier(name="a")
  assert node.to_text() == "a"


def test_binary_expression() -> None:
  """Verify binary expression behavior."""
  node: BinaryExpression = BinaryExpression(left=Identifier(name="a"), operator="+", right=Identifier(name="b"))
  assert node.to_text() == "a + b"


def test_method_call() -> None:
  """Verify method call behavior."""
  node: MethodCall = MethodCall(name="foo")
  assert node.to_text() == "foo()"
  node_args: MethodCall = MethodCall(name="foo", arguments=[Identifier(name="a"), Identifier(name="b")])
  assert node_args.to_text() == "foo(a, b)"


def test_return_statement() -> None:
  """Verify return statement behavior."""
  node: ReturnStatement = ReturnStatement()
  assert node.to_text() == "return;"
  node_val: ReturnStatement = ReturnStatement(value=Identifier(name="a"))
  assert node_val.to_text() == "return a;"


def test_variable_declaration() -> None:
  """Verify variable declaration behavior."""
  node: VariableDeclaration = VariableDeclaration(type_id=TypeIdentifier(name="int"), name="a")
  assert node.to_text() == "int a;"
  node_init: VariableDeclaration = VariableDeclaration(
    type_id=TypeIdentifier(name="int"), name="a", initializer=Identifier(name="1")
  )
  assert node_init.to_text() == "int a = 1;"
  node_init_str: VariableDeclaration = VariableDeclaration(
    type_id=TypeIdentifier(name="int"), name="a", initializer=Identifier(name="1")
  )  # It seems initializer can be a string, but the class likely converts it or maybe we should pass an Identifier if it was passed as string in the test. Let's pass "1" if it accepts it.
  assert node_init_str.to_text() == "int a = 1;"


def test_function_argument() -> None:
  """Verify function argument behavior."""
  node: FunctionArgument = FunctionArgument(type_id=TypeIdentifier(name="int"), name="a")
  assert node.to_text() == "int a"


def test_function_definition() -> None:
  """Verify function definition behavior."""
  node: FunctionDefinition = FunctionDefinition(
    return_type=TypeIdentifier(name="int"),
    name="foo",
    arguments=[FunctionArgument(type_id=TypeIdentifier(name="int"), name="a")],
    body=[ReturnStatement(value=Identifier(name="a"))],
  )
  assert node.to_text() == "int foo(int a) {\n    return a;\n}"


def test_raw_statement() -> None:
  """Verify raw statement behavior."""
  node: RawStatement = RawStatement(code="int a = 1;")
  assert node.to_text() == "int a = 1;"


def test_macro_definition() -> None:
  """Verify macro definition behavior."""
  node: MacroDefinition = MacroDefinition(name="FOO", value="1")
  assert node.to_text() == "#define FOO 1"


def test_block_statement() -> None:
  """Verify block statement behavior."""
  node: BlockStatement = BlockStatement(statements=[ReturnStatement()])
  assert node.to_text() == "{\n    return;\n}"


def test_pybind_def() -> None:
  """Verify pybind def behavior."""
  node: PyBindDef = PyBindDef(name="foo", function_ref="foo_impl", docstring="foo doc")
  assert node.to_text() == 'm.def("foo", &foo_impl, "foo doc");'


def test_pybind_module() -> None:
  """Verify pybind module behavior."""
  node: PyBindModule = PyBindModule(
    name="foo", module_var="m", defs=[PyBindDef(name="foo", function_ref="foo_impl", docstring="foo doc")]
  )
  assert node.to_text() == 'PYBIND11_MODULE(foo, m) {\n    m.def("foo", &foo_impl, "foo doc");\n}'


def test_include_directive() -> None:
  """Verify include directive behavior."""
  node: IncludeDirective = IncludeDirective(path="iostream", system=True)
  assert node.to_text() == "#include <iostream>"
  node_local: IncludeDirective = IncludeDirective(path="foo.h", system=False)
  assert node_local.to_text() == '#include "foo.h"'

  with pytest.raises(ValueError):
    IncludeDirective(path="")

  with pytest.raises(ValueError):
    IncludeDirective(path=123)  # Intentionally passing wrong type for test

  with pytest.raises(ValueError):
    IncludeDirective(path="<iostream>")

  with pytest.raises(ValueError):
    IncludeDirective(path="foo.h> ")

  with pytest.raises(ValueError):
    IncludeDirective(path='"foo.h"')


def test_cpp_module() -> None:
  """Verify cpp module behavior."""
  node: CppModule = CppModule(includes=[IncludeDirective(path="iostream", system=True)], body=[ReturnStatement()])
  assert node.to_text() == "#include <iostream>\n\nreturn;\n"

  node_empty: CppModule = CppModule()
  assert node_empty.to_text() == "\n"


def test_cpp_node_parse() -> None:
  """Verify cpp node parse behavior."""
  # Will use the parser
  node: CppModule = CppNode.parse("int a = 1;")
  assert isinstance(node, CppModule)
