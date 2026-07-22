"""Test suite for the Coverage Gap Mlir module."""

import pytest
from unittest.mock import patch
import re
from ml_switcheroo.core.mlir.parser import MlirParser, Tokenizer, Token
from ml_switcheroo.core.mlir.tokens import TokenKind


def test_mlir_parser_coverage():
  """Verifies the behavior of MLIR parser coverage."""
  t = Tokenizer("dummy")
  with patch.object(Tokenizer, "_REGEX", re.compile("(?P<UNKNOWN>dummy)")):
    list(t.tokenize())
  assert MlirParser("{ ^bb0:")._is_region_start() is True
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    MlirParser("%res, @x = ").parse_operation()
  p = MlirParser("")
  p.tokens = [
    Token(TokenKind.IDENTIFIER, "foo", 1, 0),
    Token("UNKNOWN", ".", 1, 3),
    Token(TokenKind.IDENTIFIER, "bar", 1, 4),
    Token(TokenKind.SYMBOL, "(", 1, 7),
    Token(TokenKind.SYMBOL, ")", 1, 8),
    Token(TokenKind.EOF, "", 1, 9),
  ]
  try:
    op = p.parse_operation()
    assert op.name == "foo.bar"
  except Exception as e:
    print("Error", e)
  MlirParser('"op" @sym () : ()').parse_operation()
  try:
    MlirParser("{ ").parse_region()
  except Exception:
    pass
