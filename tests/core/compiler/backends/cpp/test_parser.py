"""Unit tests for `CppParser` and `CppTransformer` (`ml_switcheroo.core.compiler.backends.cpp.parser`)."""

import pytest
from lark import Token

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  BinaryExpression,
  CppModule,
  FunctionDefinition,
  Identifier,
  MacroDefinition,
  MethodCall,
  PyBindModule,
  RawStatement,
  ReturnStatement,
  TypeIdentifier,
  VariableDeclaration,
)
from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser, CppTransformer

_NL = chr(10)


def test_cpp_parser_full_module_constructs() -> None:
  """Verify `CppParser` parses includes, macros, functions, variables, calls, raw statements, and pybind11."""
  assert CppParser(f"   {_NL} ").parse().body == []
  assert CppParser("").parse().body == []

  code = """
#include <torch/extension.h>
#include "my_header.h"

torch::Tensor forward(torch::Tensor input, torch::Tensor weights) {
    foo(1, input);
    auto result = input * weights;
    return result;
}

void empty_fn() {
    return;
}

int multi_ret() {
    return "hello";
    return 42;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "forward doc");
}
"""
  mod: CppModule = CppParser(code).parse()
  assert [(inc.path, inc.system) for inc in mod.includes] == [
    ("torch/extension.h", True),
    ("my_header.h", False),
  ]

  fwd = mod.body[0]
  assert isinstance(fwd, FunctionDefinition)
  assert fwd.name == "forward"
  assert fwd.return_type.name == "torch::Tensor"
  assert len(fwd.arguments) == 2
  assert isinstance(fwd.body[0], RawStatement)
  fwd_v = fwd.body[1]
  assert isinstance(fwd_v, VariableDeclaration)
  assert isinstance(fwd_v.initializer, BinaryExpression)
  assert isinstance(fwd.body[2], ReturnStatement)

  empty_fn = mod.body[1]
  assert isinstance(empty_fn, FunctionDefinition)
  ret0 = empty_fn.body[0]
  assert isinstance(ret0, ReturnStatement)
  assert ret0.value is None

  pyb = mod.body[3]
  assert isinstance(pyb, PyBindModule)
  assert (pyb.name, pyb.module_var) == ("TORCH_EXTENSION_NAME", "m")
  assert pyb.defs[0].name == "forward"

  # Macros, variable declarations, raw statements, and bare return at module scope
  macro_mod = CppParser(f'#define MAX_VAL 100{_NL}#define STR_MACRO "hello"{_NL}#define EMPTY_MACRO{_NL}').parse()
  m0, m1, m2 = macro_mod.body[0], macro_mod.body[1], macro_mod.body[2]
  assert isinstance(m0, MacroDefinition)
  assert (m0.name, m0.value) == ("MAX_VAL", "100")
  assert isinstance(m1, MacroDefinition)
  assert (m1.name, m1.value) == ("STR_MACRO", '"hello"')
  assert isinstance(m2, MacroDefinition)
  assert (m2.name, m2.value) == ("EMPTY_MACRO", "")

  vars_mod = CppParser('int uninit_var; float init_var = 1.0; int call_var = bar(1); int str_concat = "a" + "b";').parse()
  v0, v1, v2, v3 = vars_mod.body[0], vars_mod.body[1], vars_mod.body[2], vars_mod.body[3]
  assert isinstance(v0, VariableDeclaration)
  assert v0.initializer is None
  assert isinstance(v1, VariableDeclaration)
  assert isinstance(v1.initializer, Identifier)
  assert v1.initializer.name == "1.0"
  assert isinstance(v2, VariableDeclaration)
  assert isinstance(v2.initializer, MethodCall)
  assert v2.initializer.name == "bar"
  assert isinstance(v3, VariableDeclaration)
  assert isinstance(v3.initializer, BinaryExpression)
  assert v3.initializer.operator == "+"

  raw_stmt = CppParser('std::cout << "hello";').parse().body[0]
  assert isinstance(raw_stmt, RawStatement)
  assert raw_stmt.code == 'std::cout << "hello"'
  bare_ret = CppParser("return;").parse().body[0]
  assert isinstance(bare_ret, ReturnStatement)
  assert bare_ret.value is None


def test_cpp_parser_syntax_errors_and_transformer_fallbacks() -> None:
  """Verify `CppParser` rejects invalid syntax and `CppTransformer` handles incomplete token sequences."""
  for invalid in (
    "invalid syntax",
    "¿",
    "(",
    "int (",
    "int foo {",
    "int foo",
    "+",
    "#include foo",
    'PYBIND11_MODULE(m1, m2) { m.def("a", foo, "doc"); }',
  ):
    with pytest.raises(ValueError, match="Unexpected"):
      CppParser(invalid).parse()

  t = CppTransformer()
  with pytest.raises(AssertionError, match="No IDENTIFIER_PATH found"):
    t.include_system([Token("OTHER", "test")])
  with pytest.raises(AssertionError, match="No IDENTIFIER_PATH found"):
    t.include_local([Token("OTHER", "test")])

  mc = t.method_call([Identifier(name="func_name"), Identifier(name="a")])
  assert mc.name == "func_name"
  assert len(mc.arguments) == 1
  assert t.string_lit([Token("STRING", '"hello"')]).name == '"hello"'

  assert t.function([TypeIdentifier(name="void")]).name == ""
  assert t.func_arg([TypeIdentifier(name="int")]).name == ""
  pb = t.pybind([Token("IDENTIFIER", "mod")])
  assert (pb.name, pb.module_var) == ("mod", "")
  vd = t.var_decl_init([TypeIdentifier(name="int")])
  assert vd.name == ""
  assert vd.initializer is None
  assert t.binary_expr([Identifier(name="a"), Identifier(name="b")]).operator == ""

  from unittest.mock import MagicMock

  bad_root_parser = CppParser("int a;")
  bad_root_parser.transformer = MagicMock()
  bad_root_parser.transformer.transform.return_value = Identifier(name="not_module")
  with pytest.raises(ValueError, match="Expected CppModule root"):
    bad_root_parser.parse()
