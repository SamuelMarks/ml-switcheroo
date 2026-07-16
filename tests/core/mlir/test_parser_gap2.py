"""Auto-generated doc."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_attributes_unclosed_brace():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1 ')
  with pytest.raises(Exception):
    parser.parse()


def test_mlir_parser_region_unclosed_brace():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() ({ ')
  with pytest.raises(Exception):
    parser.parse()


def test_mlir_parser_attributes_unclosed_brace2():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() {}')
  parser.parse()


def test_mlir_parser_region_block_break():
  """Auto-generated doc."""
  parser = MlirParser("^bb0:\n")
  # Because there are no operations, it might fail syntax? The parser requires operations?
  parser.parse()


def test_mlir_parser_type_region():
  """Auto-generated doc."""
  # TokenKind.REGION_TYPE branch
  # Let's mock the token to force it
  parser = MlirParser('%0 = "foo.bar"() : () -> !foo.bar')
  # Actually just standard parsing
  parser.parse()
  # It seems to miss lines inside the loop
  parser = MlirParser('%0 = "foo.bar"() : (!foo.bar) -> !foo.bar')
  parser.parse()


def test_mlir_parser_attribute_brackets():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() { foo = [1, 2] }')
  parser.parse()
  parser = MlirParser('%0 = "foo.bar"() { foo = [] }')
  parser.parse()


def test_mlir_parser_attribute_type():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1 : i32 }')
  parser.parse()


def test_mlir_parser_attribute_comma():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1, bar = 2 }')
  parser.parse()


def test_mlir_parser_trailing_comma_attr():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1, }')
  parser.parse()


def test_mlir_parser_trailing_whitespace_val():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() { foo = "a" \n}')
  parser.parse()


def test_mlir_parser_expect_error():
  """Auto-generated doc."""
  parser = MlirParser("^bb0:\n")
  with pytest.raises(SyntaxError, match="Expected"):
    parser.expect("IDENTIFIER")


def test_mlir_parser_is_region_start():
  """Auto-generated doc."""
  parser = MlirParser('{\n %0 = "foo.bar"() : () -> () }')
  assert parser._is_region_start()


def test_mlir_parser_tokenizer_mismatch():
  """Auto-generated doc."""
  with pytest.raises(ValueError, match="Unexpected character"):
    MlirParser("%0 = \n  `")


def test_mlir_parser_peek_oob():
  """Auto-generated doc."""
  parser = MlirParser("")
  assert parser.peek(100).kind.value == "EOF"


def test_mlir_parser_op_stuck_results():
  """Auto-generated doc."""
  parser = MlirParser('foo = "foo"()')
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse()


def test_mlir_parser_is_region_start_false():
  """Auto-generated doc."""
  parser = MlirParser('{\n @SYM = "foo.bar"() : () -> () }')
  assert not parser._is_region_start()


def test_mlir_parser_op_sym_name():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() : () -> ()')
  from ml_switcheroo.core.mlir.parser import TokenKind

  parser.tokens[0].kind = TokenKind.SYM_ID
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse()


def test_mlir_parser_parse_region_nested():
  """Auto-generated doc."""
  code = '%0 = "foo.bar"() {\n ^bb0:\n %1 = "baz"() : () -> ()\n} : () -> ()'
  parser = MlirParser(code)
  parser.parse()


def test_mlir_parser_multiple_results():
  """Auto-generated doc."""
  parser = MlirParser('%0, %1 = "foo.bar"() : () -> ()')
  parser.parse()


def test_mlir_parser_op_sym_name_working():
  """Auto-generated doc."""
  parser = MlirParser('^bb0:\n @my_sym_name = "foo.bar"() : () -> ()')
  # wait it expects sym name before "foo.bar"
  parser = MlirParser('^bb0:\n @my_sym_name "foo.bar"() : () -> ()')
  parser.parse()


def test_mlir_parser_operand_sym_id():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"(@my_sym_name) : () -> ()')
  parser.parse()
