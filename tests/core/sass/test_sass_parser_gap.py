import pytest
from ml_switcheroo.compiler.frontends.sass.parser import SassParser


def test_sass_parser_unexpected_eof():
  parser = SassParser(".text")
  parser._consume()
  with pytest.raises(SyntaxError):
    parser._consume()


def test_sass_parser_expected_token_mismatch():
  parser = SassParser("MOV")
  with pytest.raises(SyntaxError):
    parser._consume(kind=1)


def test_sass_parser_bad_token():
  parser = SassParser(",")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_sass_parser_parse_line_eof():
  parser = SassParser("")
  assert parser._parse_line() is None


def test_sass_parser_operand_eof():
  parser = SassParser("MOV R0, ")


def test_sass_parser_operand_unknown():
  parser = SassParser("MOV R0, ,")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_sass_parser_directive_eof():
  parser = SassParser(".global")
  parser.parse()


def test_sass_parser_directive_eof2():
  parser = SassParser(".global")
  from unittest.mock import patch

  with patch.object(parser, "_peek", side_effect=[parser.tokens[0], parser.tokens[0], None]):
    parser.parse()


def test_sass_parser_directive_break():
  parser = SassParser(".global\n.text")
  parser.parse()

  parser = SassParser(".global\n// comment")
  parser.parse()

  parser = SassParser(".global\nlabel:")
  parser.parse()


def test_sass_parser_operand_eof2():
  parser = SassParser("MOV R0")
  from unittest.mock import patch

  with patch.object(parser, "_peek", return_value=None):
    with pytest.raises(SyntaxError, match="Unexpected EOF expecting operand"):
      parser._parse_operand()


def test_sass_parser_operand_types():
  parser = SassParser("@P0 MOV R1, R2")
  parser.parse()

  parser = SassParser("@!PT MOV R1, R2")
  parser.parse()

  parser = SassParser("MOV R0, c[0x0][0x10]")
  parser.parse()


def test_sass_parser_memory_immediate_only():
  parser = SassParser("LD R0, [0x10]")
  parser.parse()


def test_sass_parser_label_ref():
  parser = SassParser("BRA label_target")
  parser.parse()


def test_sass_parser_semicolon():
  parser = SassParser(";")
  assert parser._parse_line() is None


def test_sass_parser_directive_semicolon():
  parser = SassParser(".global main;")
  parser.parse()


def test_sass_parser_predicate_in_operand():
  parser = SassParser("@P0")
  parser._parse_operand()


def test_sass_parser_label_def_as_operand():
  parser = SassParser("label:")
  parser._parse_operand()


def test_sass_parser_memory_bank_single():
  parser = SassParser("LD R0, c[0x0]")
  parser.parse()


def test_sass_parser_memory_base_plus_offset():
  parser = SassParser("LD R0, [R1+0x10]")
  parser.parse()

  parser = SassParser("LD R0, [R1+16]")
  parser.parse()


def test_sass_parser_directive_multiline():
  parser = SassParser(".global \nmain")
  parser.parse()


def test_sass_parser_directive_comma():
  parser = SassParser(".global main, other")
  parser.parse()
