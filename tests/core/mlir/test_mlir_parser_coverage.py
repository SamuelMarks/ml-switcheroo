"""Docstring for test_mlir_parser_coverage module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_invalid_token() -> None:
  """Docstring."""
  with pytest.raises(ValueError, match="Unexpected"):
    MlirParser("~").parse()


def test_mlir_parser_sym_id() -> None:
  """Docstring."""
  code: str = "func.func @main() { return }"
  parser = MlirParser(code)
  module = parser.parse()
  assert module.body.operations[0].name == "func.func"
  assert module.body.operations[0].name_trivia[-1].text == "@main"


def test_mlir_parser_array_attr() -> None:
  """Docstring."""
  code: str = "sw.op {arr = [1, 2]}"
  parser = MlirParser(code)
  module = parser.parse()
  assert module.body.operations[0].attributes[0].value == ["1", "2"]


def test_mlir_parser_empty() -> None:
  """Docstring."""
  parser = MlirParser("   ")
  module = parser.parse()
  assert len(module.body.operations) == 0


def test_mlir_parser_op_tail_region() -> None:
  """Docstring."""
  code: str = "sw.op { ^bb0: }"
  parser = MlirParser(code)
  module = parser.parse()
  assert len(module.body.operations[0].regions) == 1


def test_mlir_parser_branch_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.core.mlir.parser import MlirTransformer

  transformer = MlirTransformer()
  op = transformer.operation([None, None])
  assert op.name == ""
  op = transformer.operation([[]])
  assert op.name == ""


def test_mlir_parser_missing_branches_cst() -> None:
  """Docstring."""
  from ml_switcheroo.core.mlir.parser import MlirParser

  code = """
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
  parser = MlirParser(code)
  module = parser.parse()
  assert len(module.body.operations) > 0


def test_mlir_parser_lexer_state_object():
  """Docstring."""
  from ml_switcheroo.core.mlir.parser import MlirLexer

  class DummyLexerState:
    """Docstring."""

    def __init__(self, text):
      """Docstring."""
      self.text = text

  ls = DummyLexerState('%0 = "dummy"()')
  lexer = MlirLexer(None)  # Assuming lexer_conf is None is ok for this
  list(lexer.lex(ls, None))


def test_mlir_parser_direct_transformer() -> None:
  """Hit transformer branches directly."""
  from ml_switcheroo.core.mlir.parser import MlirTransformer
  from lark import Tree, Token

  t = MlirTransformer()

  _c1 = Tree("op_result_list", [Tree("op_result", [Token("VAL_ID", "%0")])])
  c2 = Tree("successor_list", [Tree("successor", [Tree("caret_id", [Token("UNKNOWN", ""), Token("IDENTIFIER", "bb1")])])])
  c3 = Tree("region_list", [Tree("region", [])])

  # function_type with arrow and nested type_list_parens
  c5 = Tree(
    "function_type",
    [
      Tree("type_list_parens", [Token("TYPE", "i32")]),
      Token("ARROW", "->"),
      Tree("type_list_parens", [Token("TYPE", "f32")]),
    ],
  )

  c6 = Tree("op_tail", [Token("COLON", ":"), Tree("result_types", [Token("TYPE", "i64")])])

  c7 = Tree(
    "operands",
    [Tree("operand_list", [Tree("operand", [Tree("value_use", [Token("VAL_ID", "%0")])])])],
  )

  op = t.operation([_c1, c2, c3, c7, c5, c6])
  assert "^bb1" in op.successors


def test_mlir_parser_direct_transformer_2() -> None:
  """Docstring."""
  from ml_switcheroo.core.mlir.parser import MlirTransformer
  from lark import Tree, Token

  t = MlirTransformer()

  # op_result without nested
  _c1 = Tree("op_result_list", [Tree("op_result", [Tree("unknown", [])])])

  c5 = Tree("function_type", [Token("TYPE", "i32"), Token("ARROW", "->"), Token("TYPE", "f32")])

  op = t.operation([c5])
  assert len(op.result_types) == 1


def test_mlir_parser_direct_transformer_3() -> None:
  """Docstring."""
  from ml_switcheroo.core.mlir.parser import MlirTransformer
  from lark import Tree, Token

  t = MlirTransformer()

  c5 = Tree("function_type", [Token("TYPE", "i32"), Token("ARROW", "->"), Token("TYPE", "f32")])

  # Just mock an operation to hit 458
  _c_op = Tree("operand_list", [Tree("operand", [Token("VAL_ID", "%0")])])

  # We need an operands node to populate op.operands, so it hits arg_idx < len(op.operands)
  # operands -> operand_list -> operand -> value_use -> VAL_ID
  c_val = Tree(
    "operands",
    [Tree("operand_list", [Tree("operand", [Tree("value_use", [Token("VAL_ID", "%0")])])])],
  )

  _op = t.operation([c_val, c5])

  # 284 trailing trivia for TypeAliasDefNode
  c_alias = [
    Tree("type_alias", [Token("TYPE", "!alias")]),
    Token("EQ", "="),
    Token("TYPE", "i32"),
    Tree("trivia", [Token("WS", " ")]),
  ]
  t.type_alias_def(c_alias)
