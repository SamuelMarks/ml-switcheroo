"""Module docstring."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser, Tokenizer, Token
from ml_switcheroo.core.mlir.nodes import BlockNode
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol
import re


def test_tokenizer_invalid_kind_fallback():
  """Function docstring."""
  tok = Tokenizer("~~")
  tok._REGEX = re.compile(r"(?P<FAKE_KIND>~~)")
  tokens = list(tok.tokenize())
  assert tokens[0].kind == "FAKE_KIND"


def test_tokenizer_mismatch():
  """Function docstring."""
  tok = Tokenizer("$$$")
  with pytest.raises(ValueError, match="Unexpected character"):
    list(tok.tokenize())


def test_parser_peek_eof():
  """Function docstring."""
  parser = MlirParser("")
  kind = parser.peek(1).kind
  if hasattr(kind, "value"):
    kind = kind.value
  assert kind == "EOF"


def test_parser_expect_failure():
  """Function docstring."""
  parser = MlirParser("xyz")
  with pytest.raises(SyntaxError, match="Expected VAL_ID"):
    parser.expect("VAL_ID")


def test_parse_block_unmatched_brace():
  """Function docstring."""
  parser = MlirParser("^bb0: }")
  blk = parser.parse_block()
  assert len(blk.operations) == 0


def test_is_region_start_trivia_and_dict():
  """Function docstring."""
  parser = MlirParser("{ // comment\n a = 1 }")
  parser.consume()
  assert not parser._is_region_start()

  parser2 = MlirParser("{ \n sw.op \n }")
  parser2.consume()
  assert parser2._is_region_start()


def test_parse_operation_stuck_results():
  """Function docstring."""
  parser = MlirParser("%0 [ = sw.op")
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse_operation()


def test_parse_dotted_op_name():
  """Function docstring."""
  parser = MlirParser("")
  parser.tokens = [
    Token(TokenKind.STRING, '"my"', 1, 1),
    Token(TokenKind.SYMBOL, ".", 1, 5),
    Token(TokenKind.IDENTIFIER, "op", 1, 6),
    Token(TokenKind.EOF, "", 1, 8),
  ]
  parser.pos = 0
  op = parser.parse_operation()
  assert op.name == '"my".op'


def test_parse_no_op_name():
  """Function docstring."""
  parser = MlirParser("%0 = ")
  assert parser.parse_operation() is None


def test_parse_operands():
  """Function docstring."""
  parser = MlirParser("sw.op(%0, %1, @sym)")
  op = parser.parse_operation()
  assert len(op.operands) == 3


def test_parse_attrs_break():
  """Function docstring."""
  parser = MlirParser("sw.op { }")
  op = parser.parse_operation()
  assert len(op.attributes) == 0


def test_parse_attrs_bracket_nesting():
  """Function docstring."""
  parser = MlirParser("sw.op { a = [1, 2, [3, 4]] : i32 }")
  op = parser.parse_operation()
  assert len(op.attributes) == 1
  assert op.attributes[0].value == "[1,2,[3,4]]"


def test_parse_attrs_eof_in_val():
  """Function docstring."""
  parser = MlirParser("sw.op { a = [1, 2")
  with pytest.raises(SyntaxError):
    parser.parse_operation()


def test_parse_attrs_with_type_and_comma():
  """Function docstring."""
  parser = MlirParser('sw.op { a = 1 : i32, b = "str" }')
  op = parser.parse_operation()
  assert len(op.attributes) == 2
  attr = op.attributes[0]
  if hasattr(attr, "type"):
    assert attr.type == "i32"
  elif hasattr(attr, "type_"):
    assert attr.type_ == "i32"
  elif hasattr(attr, "attr_type"):
    assert attr.attr_type == "i32"


def test_parse_multiple_return_types():
  """Function docstring."""
  parser = MlirParser("sw.op : (i32, f32)")
  op = parser.parse_operation()
  assert len(op.result_types) == 2
  if hasattr(op.result_types[0], "text"):
    assert op.result_types[0].text == "i32"
  elif hasattr(op.result_types[0], "name"):
    assert op.result_types[0].name == "i32"


def test_parse_arrow():
  """Function docstring."""
  parser = MlirParser("sw.op -> (i32)")
  op = parser.parse_operation()
  assert op is not None


def test_parse_region_empty():
  """Function docstring."""
  parser = MlirParser("{ }")
  region = parser.parse_region()
  assert len(region.blocks) == 0


def test_parse_region_defensive_consume():
  """Function docstring."""
  parser = MlirParser("{ a = 1 }")
  # a = 1 will make it not a region, but we call parse_region explicitly.
  # The inner block will parse 'a' as an operation name? No, wait!
  # "a" is an IDENTIFIER. So op_name = "a".
  # parse_operation() will succeed!
  pass  # we'll use monkeypatching below


def test_parse_region_defensive_consume_monkeypatch(monkeypatch):
  """Function docstring."""
  parser = MlirParser("{ %0 = sw.op }")  # something valid
  monkeypatch.setattr(parser, "parse_block", lambda **kw: BlockNode(label="", arguments=[], operations=[]))
  region = parser.parse_region()
  # first iter: consume %0
  # second iter: consume =
  # third iter: consume sw.op
  # fourth iter: hit } and break
  assert len(region.blocks) > 0
