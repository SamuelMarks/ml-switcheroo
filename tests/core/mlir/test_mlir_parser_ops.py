"""Test suite for the Mlir Parser Ops module."""

import pytest
from ml_switcheroo.core.mlir.parser_ops import MlirParserOpsMixin
from ml_switcheroo.core.mlir.parser_base import MlirParserBase
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol


class DummyParser(MlirParserOpsMixin, MlirParserBase):
  """Dummy Parser class for testing purposes."""

  def _is_region_start(self):
    offset = 1
    while True:
      t = self.peek(offset)
      if t.kind in (TokenKind.WHITESPACE, TokenKind.NEWLINE, TokenKind.COMMENT):
        offset += 1
        continue
      break
    if t.kind == TokenKind.BLOCK_LABEL:
      return True
    if t.kind == TokenKind.VAL_ID:
      return True
    if t.text == Symbol.RBRACE:
      return True
    return False

  def parse_region(self):
    """Mock implementation of parse region."""
    self.consume()
    count = 0
    while not self.match(Symbol.RBRACE) and self.peek().kind != TokenKind.EOF:
      self.consume()
      count += 1
      if count > 100:
        raise RuntimeError("Stuck in dummy region parser")
    if self.match(Symbol.RBRACE):
      self.consume()
    return "RegionNodeMock"


def test_parse_op_simple():
  """Parses op simple."""
  parser = DummyParser("func.return")
  op = parser.parse_operation()
  assert op.name == "func.return"
  assert len(op.results) == 0


def test_parse_op_results():
  """Parses op results."""
  parser = DummyParser("%res = addf")
  op = parser.parse_operation()
  assert op.name == "addf"
  assert len(op.results) == 1
  assert op.results[0].name == "%res"


def test_parse_op_multi_results():
  """Parses op multi results."""
  parser = DummyParser("%r1, %r2 = multi.op")
  op = parser.parse_operation()
  assert len(op.results) == 2


def test_parse_op_stuck_results():
  """Parses op stuck results."""
  parser = DummyParser("%res foo = addf")
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse_operation()


def test_parse_op_invalid_name():
  """Parses op invalid name."""
  parser = DummyParser("%res = ")
  assert parser.parse_operation() is None


def test_parse_op_implicit_sym():
  """Parses op implicit sym."""
  parser = DummyParser("func.func @main")
  op = parser.parse_operation()
  assert op.name == "func.func"
  assert op.attributes[0].name == "sym_name"
  assert op.attributes[0].value == '"main"'


def test_parse_op_operands():
  """Parses op operands."""
  parser = DummyParser("addf(%v1, %v2)")
  op = parser.parse_operation()
  assert len(op.operands) == 2
  assert op.operands[0].name == "%v1"


def test_parse_op_operands_trailing_comma():
  """Parses op operands trailing comma."""
  parser = DummyParser("addf(%v1, %v2, )")
  op = parser.parse_operation()
  assert len(op.operands) == 2


def test_parse_op_attributes():
  """Parses op attributes."""
  parser = DummyParser('foo.op {a = 1, b = "string", c = [1, 2] : i32}')
  op = parser.parse_operation()
  assert len(op.attributes) == 3


def test_parse_op_region():
  """Parses op region."""
  parser = DummyParser("func.func { ^bb0: }")
  op = parser.parse_operation()
  assert len(op.regions) == 1


def test_parse_op_types_single():
  """Parses op types single."""
  parser = DummyParser("foo : i32")
  op = parser.parse_operation()
  assert len(op.result_types) == 1
  assert op.result_types[0].body == "i32"


def test_parse_op_types_multi():
  """Parses op types multi."""
  parser = DummyParser("foo : (i32, f32)")
  op = parser.parse_operation()
  assert len(op.result_types) == 2


def test_parse_op_types_arrow_single():
  """Parses op types arrow single."""
  parser = DummyParser("foo -> f32")
  op = parser.parse_operation()
  assert len(op.result_types) == 1


def test_parse_op_types_arrow_multi():
  """Parses op types arrow multi."""
  parser = DummyParser("foo -> (f32, i32)")
  op = parser.parse_operation()
  assert len(op.result_types) == 2


def test_parse_op_results_lookahead_limit():
  """Parses op results lookahead limit."""
  code = "%a a a a a a a a a a a a a a a a a a a a a a a a a a = op"
  parser = DummyParser(code)
  op = parser.parse_operation()
  assert op is None


def test_parse_op_attributes_eof():
  """Parses op attributes eof."""
  parser = DummyParser("foo.op {a = 1")
  with pytest.raises(SyntaxError):
    parser.parse_operation()


def test_parse_op_attributes_nested_empty():
  """Parses op attributes nested empty."""
  parser = DummyParser("foo.op {a = []}")
  op = parser.parse_operation()
  assert op.attributes[0].value == "[]"
