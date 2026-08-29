"""Test suite for the C++ CST module."""

import ast

import pytest

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  BinaryExpression,
  BlockStatement,
  CppModule,
  CppNode,
  Expression,
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
from ml_switcheroo.core.compiler.backends.cpp.mapper import ASTToCppMapper


def test_cpp_node_base() -> None:
  """Docstring."""
  node = CppNode()
  with pytest.raises(NotImplementedError):
    node.to_text()


def test_type_identifier() -> None:
  """Docstring."""
  assert TypeIdentifier("int").to_text() == "int"


def test_variable_declaration() -> None:
  """Docstring."""
  decl = VariableDeclaration(TypeIdentifier("int"), "x")
  assert decl.to_text() == "int x;"
  decl_init = VariableDeclaration(TypeIdentifier("int"), "x", "5")
  assert decl_init.to_text() == "int x = 5;"
  decl_expr = VariableDeclaration(TypeIdentifier("int"), "x", Identifier("y"))
  assert decl_expr.to_text() == "int x = y;"


def test_function_argument() -> None:
  """Docstring."""
  arg = FunctionArgument(TypeIdentifier("float"), "y")
  assert arg.to_text() == "float y"


def test_function_definition() -> None:
  """Docstring."""
  func = FunctionDefinition(
    return_type=TypeIdentifier("void"),
    name="my_func",
    arguments=[FunctionArgument(TypeIdentifier("int"), "x")],
    body=[RawStatement("x++;")],
  )
  text = func.to_text()
  assert "void my_func(int x) {" in text
  assert "    x++;" in text
  assert text.endswith("}")


def test_macro_definition() -> None:
  """Docstring."""
  macro = MacroDefinition("MAX_SIZE", "100")
  assert macro.to_text() == "#define MAX_SIZE 100"


def test_block_statement() -> None:
  """Docstring."""
  block = BlockStatement([RawStatement("int x = 0;"), RawStatement("x++;")])
  text = block.to_text()
  assert "{\n    int x = 0;\n    x++;\n}" in text


def test_pybind_def() -> None:
  """Docstring."""
  pdef = PyBindDef("my_func", "cpp_my_func", "docstring")
  assert pdef.to_text() == 'm.def("my_func", &cpp_my_func, "docstring");'


def test_pybind_module() -> None:
  """Docstring."""
  pmod = PyBindModule("TORCH_EXTENSION_NAME", "m", [PyBindDef("f", "f_cpp", "doc")])
  text = pmod.to_text()
  assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {" in text
  assert '    m.def("f", &f_cpp, "doc");' in text
  assert text.endswith("}")


def test_include_directive() -> None:
  """Docstring."""
  inc1 = IncludeDirective("iostream", system=True)
  assert inc1.to_text() == "#include <iostream>"
  inc2 = IncludeDirective("my_header.h")
  assert inc2.to_text() == '#include "my_header.h"'


def test_cpp_module_empty() -> None:
  """Docstring."""
  mod = CppModule(includes=[], body=[MacroDefinition("A", "1")])
  text = mod.to_text()
  assert "#define A 1" in text
  assert "#include" not in text


def test_cpp_module() -> None:
  """Docstring."""
  mod = CppModule(includes=[IncludeDirective("iostream", system=True)], body=[MacroDefinition("A", "1")])
  text = mod.to_text()
  assert "#include <iostream>" in text
  assert "#define A 1" in text


def test_expressions() -> None:
  """Docstring."""
  i = Identifier("foo")
  assert i.to_text() == "foo"

  b = BinaryExpression(Identifier("a"), "+", Identifier("b"))
  assert b.to_text() == "a + b"

  m = MethodCall("printf", [Identifier("a")])
  assert m.to_text() == "printf(a)"

  r1 = ReturnStatement()
  assert r1.to_text() == "return;"

  r2 = ReturnStatement(Identifier("a"))
  assert r2.to_text() == "return a;"


# --- Merged from test_cst_extra.py ---


def test_cst_not_implemented() -> None:
  """Docstring."""

  class Dummy(CppNode):
    """Dummy class."""

    pass

  with pytest.raises(NotImplementedError):
    Dummy().to_text()


def test_include_directive_validation() -> None:
  """Docstring."""
  with pytest.raises(ValueError):
    IncludeDirective(path=123)  # type: ignore
  with pytest.raises(ValueError):
    IncludeDirective(path="<foo>")


def test_cst_parse_method() -> None:
  """Docstring."""
  code: str = "int x = 5;"
  node: CppNode = CppNode.parse(code)
  assert isinstance(node, CppModule)


def test_ast_mapper_call_attr() -> None:
  """Docstring."""
  # Covers line 37 in mapper.py (naive attribute translation)
  mapper = ASTToCppMapper()
  tree: ast.expr = ast.parse("a.b.c()", mode="eval").body
  cpp_expr: Expression = mapper.map_expression(tree)
  assert cpp_expr.to_text() == "c()"
