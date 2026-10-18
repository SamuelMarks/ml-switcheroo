"""Docstring for test_mlir_parser_coverage module."""

import typing

import pytest

from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_invalid_token() -> None:
  """Verify mlir parser invalid token behavior."""
  with pytest.raises(ValueError, match="Unexpected"):
    MlirParser("~").parse()


def test_mlir_parser_sym_id() -> None:
  """Verify mlir parser sym id behavior."""
  code: str = "func.func @main() { return }"
  parser: typing.Any = MlirParser(code)
  module: typing.Any = parser.parse()
  assert module.body.operations[0].name == "func.func"
  assert module.body.operations[0].name_trivia[-1].text == "@main"


def test_mlir_parser_array_attr() -> None:
  """Verify mlir parser array attr behavior."""
  code: str = "sw.op {arr = [1, 2]}"
  parser: typing.Any = MlirParser(code)
  module: typing.Any = parser.parse()
  assert module.body.operations[0].attributes[0].value == ["1", "2"]


def test_mlir_parser_empty() -> None:
  """Verify mlir parser empty behavior."""
  parser: typing.Any = MlirParser("   ")
  module: typing.Any = parser.parse()
  assert len(module.body.operations) == 0


def test_mlir_parser_op_tail_region() -> None:
  """Verify mlir parser op tail region behavior."""
  code: str = "sw.op { ^bb0: }"
  parser: typing.Any = MlirParser(code)
  module: typing.Any = parser.parse()
  assert len(module.body.operations[0].regions) == 1


def test_mlir_parser_branch_coverage() -> None:
  """Verify mlir parser branch coverage behavior."""
  from ml_switcheroo.core.mlir.parser import MlirTransformer

  transformer: typing.Any = MlirTransformer()
  op: typing.Any = transformer.operation([None, None])
  assert op.name == ""
  op = transformer.operation([[]])
  assert op.name == ""


def test_mlir_parser_missing_branches_cst() -> None:
  """Verify mlir parser missing branches cst behavior."""
  from ml_switcheroo.core.mlir.parser import MlirParser

  code: typing.Any = """
  %0, %1 = "my.op"()
  "my.br"()[^bb1]
  "my.op"() ( { } )
  "my.op"() {a = 1}
  "my.op"(%0, %1) : (i32, i64) -> (f32, f64)
  "my.op"(%0) : i32 -> f32
  "my.op"() loc("foo.py")
  "my.op"() : () -> ()
  "my.op"() -> i32
  """
  parser: typing.Any = MlirParser(code)
  module: typing.Any = parser.parse()
  assert len(module.body.operations) > 0


def test_mlir_parser_lexer_state_object() -> None:
  """Verify mlir parser lexer state object behavior."""
  from ml_switcheroo.core.mlir.parser import MlirLexer

  class DummyLexerState:
    """Test helper class DummyLexerState."""

    def __init__(self, text: typing.Any) -> None:
      """Initialize test helper instance.

      Args:
          text: Test argument or fixture.
      """
      self.text = text

  ls: typing.Any = DummyLexerState('%0 = "dummy"()')
  lexer: typing.Any = MlirLexer(None)  # Assuming lexer_conf is None is ok for this
  _res: typing.Any = list(lexer.lex(ls, None))
  assert len(_res) > 0
  assert _res[0].value == "%0"


def test_mlir_parser_direct_transformer() -> None:
  """Hit transformer branches directly."""
  from lark import Token, Tree

  from ml_switcheroo.core.mlir.parser import MlirTransformer

  t: typing.Any = MlirTransformer()

  _c1: typing.Any = Tree("op_result_list", [Tree("op_result", [Token("VAL_ID", "%0")])])
  c2: typing.Any = Tree(
    "successor_list", [Tree("successor", [Tree("caret_id", [Token("UNKNOWN", ""), Token("IDENTIFIER", "bb1")])])]
  )
  c3: typing.Any = Tree("region_list", [Tree("region", [])])

  # function_type with arrow and nested type_list_parens
  c5: typing.Any = Tree(
    "function_type",
    [
      Tree("type_list_parens", [Token("TYPE", "i32")]),
      Token("ARROW", "->"),
      Tree("type_list_parens", [Token("TYPE", "f32")]),
    ],
  )

  c6: typing.Any = Tree("op_tail", [Token("COLON", ":"), Tree("result_types", [Token("TYPE", "i64")])])

  c7: typing.Any = Tree(
    "operands",
    [Tree("operand_list", [Tree("operand", [Tree("value_use", [Token("VAL_ID", "%0")])])])],
  )

  op: typing.Any = t.operation([_c1, c2, c3, c7, c5, c6])
  assert "^bb1" in op.successors


def test_mlir_parser_direct_transformer_2() -> None:
  """Verify mlir parser direct transformer 2 behavior."""
  from lark import Token, Tree

  from ml_switcheroo.core.mlir.parser import MlirTransformer

  t: typing.Any = MlirTransformer()

  # op_result without nested
  _c1: typing.Any = Tree("op_result_list", [Tree("op_result", [Tree("unknown", [])])])

  c5: typing.Any = Tree("function_type", [Token("TYPE", "i32"), Token("ARROW", "->"), Token("TYPE", "f32")])

  op: typing.Any = t.operation([c5])
  assert len(op.result_types) == 1


def test_mlir_parser_direct_transformer_3() -> None:
  """Verify mlir parser direct transformer 3 behavior."""
  from lark import Token, Tree

  from ml_switcheroo.core.mlir.parser import MlirTransformer

  t: typing.Any = MlirTransformer()

  c5: typing.Any = Tree("function_type", [Token("TYPE", "i32"), Token("ARROW", "->"), Token("TYPE", "f32")])

  # Just mock an operation to hit 458
  _c_op: typing.Any = Tree("operand_list", [Tree("operand", [Token("VAL_ID", "%0")])])

  # We need an operands node to populate op.operands, so it hits arg_idx < len(op.operands)
  # operands -> operand_list -> operand -> value_use -> VAL_ID
  c_val: typing.Any = Tree(
    "operands",
    [Tree("operand_list", [Tree("operand", [Tree("value_use", [Token("VAL_ID", "%0")])])])],
  )

  _op: typing.Any = t.operation([c_val, c5])

  # 284 trailing trivia for TypeAliasDefNode
  c_alias: typing.Any = [
    Tree("type_alias", [Token("TYPE", "!alias")]),
    Token("EQ", "="),
    Token("TYPE", "i32"),
    Tree("trivia", [Token("WS", " ")]),
  ]
  _res: typing.Any = t.type_alias_def(c_alias)
  assert _res.name == "alias"
  assert _res.type_node.body == "i32"
