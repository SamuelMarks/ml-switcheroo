"""Test suite for the Parser Gap2 module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_attributes_unclosed_brace():
  """Verifies the behavior of MLIR parser attributes unclosed brace."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1 ')
  with pytest.raises(Exception):
    parser.parse()


def test_mlir_parser_region_unclosed_brace():
  """Verifies the behavior of MLIR parser region unclosed brace."""
  parser = MlirParser('%0 = "foo.bar"() ({ ')
  with pytest.raises(Exception):
    parser.parse()


def test_mlir_parser_attributes_unclosed_brace2():
  """Verifies the behavior of MLIR parser attributes unclosed brace2."""
  parser = MlirParser('%0 = "foo.bar"() {}')
  parser.parse()


def test_mlir_parser_region_block_break():
  """Verifies the behavior of MLIR parser region block break."""
  parser = MlirParser("^bb0:\n")
  parser.parse()


def test_mlir_parser_type_region():
  """Verifies the behavior of MLIR parser type region."""
  parser = MlirParser('%0 = "foo.bar"() : () -> !foo.bar')
  parser.parse()
  parser = MlirParser('%0 = "foo.bar"() : (!foo.bar) -> !foo.bar')
  parser.parse()


def test_mlir_parser_attribute_brackets():
  """Verifies the behavior of MLIR parser attribute brackets."""
  parser = MlirParser('%0 = "foo.bar"() { foo = [1, 2] }')
  parser.parse()
  parser = MlirParser('%0 = "foo.bar"() { foo = [] }')
  parser.parse()


def test_mlir_parser_attribute_type():
  """Verifies the behavior of MLIR parser attribute type."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1 : i32 }')
  parser.parse()


def test_mlir_parser_attribute_comma():
  """Verifies the behavior of MLIR parser attribute comma."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1, bar = 2 }')
  parser.parse()


def test_mlir_parser_trailing_comma_attr():
  """Verifies the behavior of MLIR parser trailing comma attribute."""
  parser = MlirParser('%0 = "foo.bar"() { foo = 1, }')
  parser.parse()


def test_mlir_parser_trailing_whitespace_val():
  """Verifies the behavior of MLIR parser trailing whitespace value."""
  parser = MlirParser('%0 = "foo.bar"() { foo = "a" \n}')
  parser.parse()


def test_mlir_parser_expect_error():
  """Verifies the behavior of MLIR parser expect correctly handling an error."""
  parser = MlirParser("^bb0:\n")
  with pytest.raises(SyntaxError, match="Expected"):
    parser.expect("IDENTIFIER")


def test_mlir_parser_is_region_start():
  """Verifies the behavior of MLIR parser is region start."""
  parser = MlirParser('{\n %0 = "foo.bar"() : () -> () }')
  assert parser._is_region_start()


def test_mlir_parser_tokenizer_mismatch():
  """Verifies the behavior of MLIR parser tokenizer mismatch."""
  with pytest.raises(ValueError, match="Unexpected character"):
    MlirParser("%0 = \n  `")


def test_mlir_parser_peek_oob():
  """Verifies the behavior of MLIR parser peek oob."""
  parser = MlirParser("")
  assert parser.peek(100).kind.value == "EOF"


def test_mlir_parser_op_stuck_results():
  """Verifies the behavior of MLIR parser op stuck results."""
  parser = MlirParser('foo = "foo"()')
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse()


def test_mlir_parser_is_region_start_false():
  """Verifies the behavior of MLIR parser is region start false."""
  parser = MlirParser('{\n @SYM = "foo.bar"() : () -> () }')
  assert not parser._is_region_start()


def test_mlir_parser_op_sym_name():
  """Verifies the behavior of MLIR parser op sym name."""
  parser = MlirParser('%0 = "foo.bar"() : () -> ()')
  from ml_switcheroo.core.mlir.parser import TokenKind

  parser.tokens[0].kind = TokenKind.SYM_ID
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse()


def test_mlir_parser_parse_region_nested():
  """Verifies the behavior of MLIR parser parse region nested."""
  code = '%0 = "foo.bar"() {\n ^bb0:\n %1 = "baz"() : () -> ()\n} : () -> ()'
  parser = MlirParser(code)
  parser.parse()


def test_mlir_parser_multiple_results():
  """Verifies the behavior of MLIR parser multiple results."""
  parser = MlirParser('%0, %1 = "foo.bar"() : () -> ()')
  parser.parse()


def test_mlir_parser_op_sym_name_working():
  """Verifies the behavior of MLIR parser op sym name working."""
  parser = MlirParser('^bb0:\n @my_sym_name = "foo.bar"() : () -> ()')
  parser = MlirParser('^bb0:\n @my_sym_name "foo.bar"() : () -> ()')
  parser.parse()


def test_mlir_parser_operand_sym_id():
  """Verifies the behavior of MLIR parser operand sym id."""
  parser = MlirParser('%0 = "foo.bar"(@my_sym_name) : () -> ()')
  parser.parse()
