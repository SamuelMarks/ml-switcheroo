import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_attributes_unclosed_brace():
  parser = MlirParser('%0 = "foo.bar"() { foo = 1 ')
  with pytest.raises(Exception):
    parser.parse()


def test_mlir_parser_region_unclosed_brace():
  parser = MlirParser('%0 = "foo.bar"() ({ ')
  with pytest.raises(Exception):
    parser.parse()


def test_mlir_parser_attributes_unclosed_brace2():
  parser = MlirParser('%0 = "foo.bar"() {}')
  parser.parse()


def test_mlir_parser_region_block_break():
  parser = MlirParser("^bb0:\n")
  # Because there are no operations, it might fail syntax? The parser requires operations?
  ast = parser.parse()


def test_mlir_parser_type_region():
  # TokenKind.REGION_TYPE branch
  # Let's mock the token to force it
  parser = MlirParser('%0 = "foo.bar"() : () -> !foo.bar')
  # Actually just standard parsing
  ast = parser.parse()
  # It seems to miss lines inside the loop
  parser = MlirParser('%0 = "foo.bar"() : (!foo.bar) -> !foo.bar')
  parser.parse()


def test_mlir_parser_attribute_brackets():
  parser = MlirParser('%0 = "foo.bar"() { foo = [1, 2] }')
  parser.parse()
  parser = MlirParser('%0 = "foo.bar"() { foo = [] }')
  parser.parse()


def test_mlir_parser_attribute_type():
  parser = MlirParser('%0 = "foo.bar"() { foo = 1 : i32 }')
  parser.parse()


def test_mlir_parser_attribute_comma():
  parser = MlirParser('%0 = "foo.bar"() { foo = 1, bar = 2 }')
  parser.parse()


def test_mlir_parser_trailing_comma_attr():
  parser = MlirParser('%0 = "foo.bar"() { foo = 1, }')
  parser.parse()


def test_mlir_parser_trailing_whitespace_val():
  parser = MlirParser('%0 = "foo.bar"() { foo = "a" \n}')
  parser.parse()


def test_mlir_parser_expect_error():
  parser = MlirParser("^bb0:\n")
  with pytest.raises(SyntaxError, match="Expected"):
    parser.expect("IDENTIFIER")


def test_mlir_parser_is_region_start():
  parser = MlirParser('{\n %0 = "foo.bar"() : () -> () }')
  assert parser._is_region_start() == True


def test_mlir_parser_tokenizer_mismatch():
  with pytest.raises(ValueError, match="Unexpected character"):
    MlirParser("%0 = \n  `")


def test_mlir_parser_peek_oob():
  parser = MlirParser("")
  assert parser.peek(100).kind.value == "EOF"


def test_mlir_parser_op_stuck_results():
  parser = MlirParser('foo = "foo"()')
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse()


def test_mlir_parser_is_region_start_false():
  parser = MlirParser('{\n @SYM = "foo.bar"() : () -> () }')
  assert parser._is_region_start() == False


def test_mlir_parser_op_sym_name():
  parser = MlirParser('%0 = "foo.bar"() : () -> ()')
  from ml_switcheroo.core.mlir.parser import TokenKind

  parser.tokens[0].kind = TokenKind.SYM_ID
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    parser.parse()


def test_mlir_parser_parse_region_nested():
  code = '%0 = "foo.bar"() {\n ^bb0:\n %1 = "baz"() : () -> ()\n} : () -> ()'
  parser = MlirParser(code)
  parser.parse()


def test_mlir_parser_multiple_results():
  parser = MlirParser('%0, %1 = "foo.bar"() : () -> ()')
  parser.parse()


def test_mlir_parser_op_sym_name_working():
  parser = MlirParser('^bb0:\n @my_sym_name = "foo.bar"() : () -> ()')
  # wait it expects sym name before "foo.bar"
  parser = MlirParser('^bb0:\n @my_sym_name "foo.bar"() : () -> ()')
  parser.parse()


def test_mlir_parser_operand_sym_id():
  parser = MlirParser('%0 = "foo.bar"(@my_sym_name) : () -> ()')
  parser.parse()
