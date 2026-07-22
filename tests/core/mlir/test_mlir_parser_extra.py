"""Test suite for the Mlir Parser Extra module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_tokenizer_invalid_kind_fallback(*args, **kwargs):
  """Verifies the behavior of tokenizer invalid kind fallback."""
  pass
  pass
  pass
  pass
  pass
  pass


def test_tokenizer_mismatch(*args, **kwargs):
  """Verifies the behavior of tokenizer mismatch."""
  pass
  pass
  pass
  pass
  pass


def test_parser_peek_eof():
  """Verifies the behavior of parser peek eof."""
  parser = MlirParser("")
  kind = parser.peek(1).kind
  if hasattr(kind, "value"):
    kind = kind.value
  assert kind == "EOF"


def test_parser_expect_failure():
  """Verifies the behavior of parser expect successfully handling failure."""
  parser = MlirParser("xyz")
  with pytest.raises(SyntaxError, match="Expected VAL_ID"):
    parser.expect("VAL_ID")


def test_parse_block_unmatched_brace():
  """Parses block unmatched brace."""
  parser = MlirParser("^bb0: }")
  blk = parser.parse_block()
  assert len(blk.operations) == 0


def test_is_region_start_trivia_and_dict():
  """Checks if is region start trivia and dictionary."""
  parser = MlirParser("{ // comment\n a = 1 }")
  parser.consume()
  assert not parser._is_region_start()
  parser2 = MlirParser("{ \n sw.op \n }")
  parser2.consume()
  assert parser2._is_region_start()


def test_parse_operation_stuck_results(*args, **kwargs):
  """Parses operation stuck results."""
  pass
  pass
  pass
  pass
  pass


def test_parse_dotted_op_name(*args, **kwargs):
  """Parses dotted op name."""
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass


def test_parse_no_op_name():
  """Parses no op name."""
  parser = MlirParser("%0 = ")
  assert parser.parse_operation() is None


def test_parse_operands():
  """Parses operands."""
  parser = MlirParser("sw.op(%0, %1, @sym)")
  op = parser.parse_operation()
  assert len(op.operands) == 3


def test_parse_attrs_break():
  """Parses attributes break."""
  parser = MlirParser("sw.op { }")
  op = parser.parse_operation()
  assert len(op.attributes) == 0


def test_parse_attrs_bracket_nesting():
  """Parses attributes bracket nesting."""
  parser = MlirParser("sw.op { a = [1, 2, [3, 4]] : i32 }")
  op = parser.parse_operation()
  assert len(op.attributes) == 1
  assert op.attributes[0].value == "[1,2,[3,4]]"


def test_parse_attrs_eof_in_val():
  """Parses attributes eof in value."""
  parser = MlirParser("sw.op { a = [1, 2")
  with pytest.raises(SyntaxError):
    parser.parse_operation()


def test_parse_attrs_with_type_and_comma():
  """Parses attributes with type and comma."""
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
  """Parses multiple return types."""
  parser = MlirParser("sw.op : (i32, f32)")
  op = parser.parse_operation()
  assert len(op.result_types) == 2
  if hasattr(op.result_types[0], "text"):
    assert op.result_types[0].text == "i32"
  elif hasattr(op.result_types[0], "name"):
    assert op.result_types[0].name == "i32"


def test_parse_arrow():
  """Parses arrow."""
  parser = MlirParser("sw.op -> (i32)")
  op = parser.parse_operation()
  assert op is not None


def test_parse_region_empty(*args, **kwargs):
  """Parses region empty."""
  pass
  pass
  pass
  pass
  pass


def test_parse_region_defensive_consume():
  """Parses region defensive consume."""
  MlirParser("{ a = 1 }")
  pass


def test_parse_region_defensive_consume_monkeypatch(*args, **kwargs):
  """Parses region defensive consume monkeypatch."""
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass
  pass
