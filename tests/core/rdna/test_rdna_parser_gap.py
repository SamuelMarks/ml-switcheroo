"""Test suite for the Rdna Parser Gap module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser


def test_rdna_parser_unexpected_eof():
  """Verifies the behavior of RDNA parser unexpected eof."""
  parser = RdnaParser(".text")
  parser._consume()
  with pytest.raises(SyntaxError, match="Unexpected End of File"):
    parser._consume()


def test_rdna_parser_expected_token_mismatch():
  """Verifies the behavior of RDNA parser expected token mismatch."""
  parser = RdnaParser("v_add_f32")
  with pytest.raises(SyntaxError, match="Expected"):
    parser._consume(kind=1)


def test_rdna_parser_parse_line_eof():
  """Verifies the behavior of RDNA parser parse line eof."""
  parser = RdnaParser("")
  assert parser._parse_line() is None


def test_rdna_parser_bad_token():
  """Verifies the behavior of RDNA parser bad token."""
  parser = RdnaParser(",")
  with pytest.raises(SyntaxError, match="Unexpected token"):
    parser.parse()


def test_rdna_parser_directive_eof():
  """Verifies the behavior of RDNA parser directive eof."""
  parser = RdnaParser(".amdgcn_target")
  parser.parse()


def test_rdna_parser_instruction_eof():
  """Verifies the behavior of RDNA parser instruction eof."""
  parser = RdnaParser("v_add_f32")
  parser.parse()


def test_rdna_parser_operand_eof():
  """Verifies the behavior of RDNA parser operand eof."""
  RdnaParser("v_add_f32 ")


def test_rdna_parser_special_reg():
  """Verifies the behavior of RDNA parser special reg."""
  parser = RdnaParser("v_add_f32 exec")
  parser.parse()


def test_rdna_parser_coverage_remaining():
  """Verifies the behavior of RDNA parser coverage remaining."""
  parser = RdnaParser(".directive")
  parser.parse()
  parser = RdnaParser(".directive\nparam")
  parser.parse()
  parser = RdnaParser("v_add_f32 .directive")
  parser.parse()
  parser = RdnaParser("v_add_f32 ,")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_rdna_parser_directive_eof2():
  """Verifies the behavior of RDNA parser directive eof2."""
  parser = RdnaParser(".directive")
  parser.parse()


def test_rdna_parser_operand_eof2():
  """Verifies the behavior of RDNA parser operand eof2."""
  parser = RdnaParser("v_add_f32 v0")
  from unittest.mock import patch

  with patch.object(parser, "_peek", return_value=None):
    with pytest.raises(SyntaxError, match="Unexpected EOF expecting operand"):
      parser._parse_operand()
