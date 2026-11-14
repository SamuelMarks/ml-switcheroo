"""Unit tests for C++ CST nodes (`ml_switcheroo.core.compiler.backends.cpp.cst`)."""

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

_NL = chr(10)


def test_cpp_cst_nodes_serialization_and_validation() -> None:
  """Verify `to_text()` across all C++ CST node classes and `IncludeDirective` delimiter validation."""
  with pytest.raises(NotImplementedError):
    CppNode().to_text()

  class _UnimplementedSubclass(CppNode):
    """Subclass inheriting abstract `to_text()`."""

  with pytest.raises(NotImplementedError):
    _UnimplementedSubclass().to_text()

  assert TypeIdentifier("int").to_text() == "int"
  assert Identifier("foo").to_text() == "foo"
  assert BinaryExpression(Identifier("a"), "+", Identifier("b")).to_text() == "a + b"
  assert MethodCall("foo").to_text() == "foo()"
  assert MethodCall("foo", [Identifier("a"), Identifier("b")]).to_text() == "foo(a, b)"
  assert ReturnStatement().to_text() == "return;"
  assert ReturnStatement(Identifier("a")).to_text() == "return a;"

  assert VariableDeclaration(TypeIdentifier("int"), "x").to_text() == "int x;"
  assert VariableDeclaration(TypeIdentifier("int"), "x", "5").to_text() == "int x = 5;"
  assert VariableDeclaration(TypeIdentifier("int"), "x", Identifier("y")).to_text() == "int x = y;"
  assert FunctionArgument(TypeIdentifier("float"), "y").to_text() == "float y"
  assert RawStatement("int a = 1;").to_text() == "int a = 1;"
  assert MacroDefinition("MAX_SIZE", "100").to_text() == "#define MAX_SIZE 100"

  fn = FunctionDefinition(
    return_type=TypeIdentifier("int"),
    name="foo",
    arguments=[FunctionArgument(TypeIdentifier("int"), "a")],
    body=[ReturnStatement(Identifier("a"))],
  )
  assert fn.to_text() == f"int foo(int a) {{{_NL}    return a;{_NL}}}"
  assert (
    BlockStatement([RawStatement("int x = 0;"), RawStatement("x++;")]).to_text()
    == f"{{{_NL}    int x = 0;{_NL}    x++;{_NL}}}"
  )

  pdef = PyBindDef("foo", "foo_impl", "foo doc")
  assert pdef.to_text() == 'm.def("foo", &foo_impl, "foo doc");'
  pmod = PyBindModule("foo", "m", [pdef])
  assert pmod.to_text() == f'PYBIND11_MODULE(foo, m) {{{_NL}    m.def("foo", &foo_impl, "foo doc");{_NL}}}'

  assert IncludeDirective("iostream", system=True).to_text() == "#include <iostream>"
  assert IncludeDirective("foo.h", system=False).to_text() == '#include "foo.h"'

  for empty_path in ("", "   "):
    with pytest.raises(ValueError, match="Include path must be a non-empty string"):
      IncludeDirective(path=empty_path)

  for bad_delim in ("<iostream>", "foo.h> ", '"foo.h"'):
    with pytest.raises(ValueError, match="Include path must not contain delimiters"):
      IncludeDirective(path=bad_delim)

  assert CppModule().to_text() == _NL
  assert CppModule(includes=[], body=[MacroDefinition("A", "1")]).to_text() == f"#define A 1{_NL}"
  mod_full = CppModule(includes=[IncludeDirective("iostream", system=True)], body=[ReturnStatement()])
  assert mod_full.to_text() == f"#include <iostream>{_NL}{_NL}return;{_NL}"

  parsed = CppNode.parse("int a = 1;")
  assert isinstance(parsed, CppModule)
