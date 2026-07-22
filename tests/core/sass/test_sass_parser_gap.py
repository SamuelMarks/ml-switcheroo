"""Test suite for the Sass Parser Gap module."""

import pytest
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser


def test_sass_parser_unexpected_eof():
  """Verifies the behavior of SASS parser unexpected eof."""
  parser = SassParser(".text")
  parser._consume()
  with pytest.raises(SyntaxError):
    parser._consume()


def test_sass_parser_expected_token_mismatch():
  """Verifies the behavior of SASS parser expected token mismatch."""
  parser = SassParser("MOV")
  with pytest.raises(SyntaxError):
    parser._consume(kind=1)


def test_sass_parser_bad_token():
  """Verifies the behavior of SASS parser bad token."""
  parser = SassParser(",")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_sass_parser_parse_line_eof():
  """Verifies the behavior of SASS parser parse line eof."""
  parser = SassParser("")
  assert parser._parse_line() is None


def test_sass_parser_operand_eof():
  """Verifies the behavior of SASS parser operand eof."""
  SassParser("MOV R0, ")


def test_sass_parser_operand_unknown():
  """Verifies the behavior of SASS parser operand unknown."""
  parser = SassParser("MOV R0, ,")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_sass_parser_directive_eof():
  """Verifies the behavior of SASS parser directive eof."""
  parser = SassParser(".global")
  parser.parse()


def test_sass_parser_directive_eof2():
  """Verifies the behavior of SASS parser directive eof2."""
  parser = SassParser(".global")
  from unittest.mock import patch

  with patch.object(parser, "_peek", side_effect=[parser.tokens[0], parser.tokens[0], None]):
    parser.parse()


def test_sass_parser_directive_break():
  """Verifies the behavior of SASS parser directive break."""
  parser = SassParser(".global\n.text")
  parser.parse()
  parser = SassParser(".global\n// comment")
  parser.parse()
  parser = SassParser(".global\nlabel:")
  parser.parse()


def test_sass_parser_operand_eof2():
  """Verifies the behavior of SASS parser operand eof2."""
  parser = SassParser("MOV R0")
  from unittest.mock import patch

  with patch.object(parser, "_peek", return_value=None):
    with pytest.raises(SyntaxError, match="Unexpected EOF expecting operand"):
      parser._parse_operand()


def test_sass_parser_operand_types():
  """Verifies the behavior of SASS parser operand types."""
  parser = SassParser("@P0 MOV R1, R2")
  parser.parse()
  parser = SassParser("@!PT MOV R1, R2")
  parser.parse()
  parser = SassParser("MOV R0, c[0x0][0x10]")
  parser.parse()


def test_sass_parser_memory_immediate_only():
  """Verifies the behavior of SASS parser memory immediate only."""
  parser = SassParser("LD R0, [0x10]")
  parser.parse()


def test_sass_parser_label_ref():
  """Verifies the behavior of SASS parser label reference."""
  parser = SassParser("BRA label_target")
  parser.parse()


def test_sass_parser_semicolon():
  """Verifies the behavior of SASS parser semicolon."""
  parser = SassParser(";")
  assert parser._parse_line() is None


def test_sass_parser_directive_semicolon():
  """Verifies the behavior of SASS parser directive semicolon."""
  parser = SassParser(".global main;")
  parser.parse()


def test_sass_parser_predicate_in_operand():
  """Verifies the behavior of SASS parser predicate in operand."""
  parser = SassParser("@P0")
  parser._parse_operand()


def test_sass_parser_label_def_as_operand():
  """Verifies the behavior of SASS parser label def as operand."""
  parser = SassParser("label:")
  parser._parse_operand()


def test_sass_parser_memory_bank_single():
  """Verifies the behavior of SASS parser memory bank single."""
  parser = SassParser("LD R0, c[0x0]")
  parser.parse()


def test_sass_parser_memory_base_plus_offset():
  """Verifies the behavior of SASS parser memory base plus offset."""
  parser = SassParser("LD R0, [R1+0x10]")
  parser.parse()
  parser = SassParser("LD R0, [R1+16]")
  parser.parse()


def test_sass_parser_directive_multiline():
  """Verifies the behavior of SASS parser directive multiline."""
  parser = SassParser(".global \nmain")
  parser.parse()


def test_sass_parser_directive_comma():
  """Verifies the behavior of SASS parser directive comma."""
  parser = SassParser(".global main, other")
  parser.parse()
