import pytest
from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser


def test_rdna_parser_unexpected_eof():
  parser = RdnaParser(".text")
  # Manually drain
  parser._consume()
  with pytest.raises(SyntaxError, match="Unexpected End of File"):
    parser._consume()


def test_rdna_parser_expected_token_mismatch():
  parser = RdnaParser("v_add_f32")
  with pytest.raises(SyntaxError, match="Expected"):
    parser._consume(kind=1)  # Using a wrong enum if it's enum


def test_rdna_parser_parse_line_eof():
  parser = RdnaParser("")
  assert parser._parse_line() is None


def test_rdna_parser_bad_token():
  parser = RdnaParser(",")
  with pytest.raises(SyntaxError, match="Unexpected token"):
    parser.parse()


def test_rdna_parser_directive_eof():
  parser = RdnaParser(".amdgcn_target")
  parser.parse()


def test_rdna_parser_instruction_eof():
  parser = RdnaParser("v_add_f32")
  ast = parser.parse()


def test_rdna_parser_operand_eof():
  parser = RdnaParser("v_add_f32 ")


def test_rdna_parser_special_reg():
  parser = RdnaParser("v_add_f32 exec")
  ast = parser.parse()


def test_rdna_parser_coverage_remaining():
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
  parser = RdnaParser(".directive")
  # if we force `next_t` to evaluate to False inside the while loop
  # We can patch `_peek` to return None specifically inside _parse_directive
  from unittest.mock import patch

  # `_parse_directive` calls `_consume` then loops
  # token list: [DIRECTIVE(.directive)]
  # inside loop, `_peek()` returns `None`
  parser.parse()


def test_rdna_parser_operand_eof2():
  parser = RdnaParser("v_add_f32 v0")
  # if we force `_parse_operand` to receive None from `_peek`
  from unittest.mock import patch

  with patch.object(parser, "_peek", return_value=None):
    with pytest.raises(SyntaxError, match="Unexpected EOF expecting operand"):
      parser._parse_operand()
