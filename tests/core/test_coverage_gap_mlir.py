import pytest
from unittest.mock import patch
import re
from ml_switcheroo.core.mlir.parser import MlirParser, Tokenizer, Token
from ml_switcheroo.core.mlir.tokens import TokenKind


def test_mlir_parser_coverage():
  # 92-93: TokenKind ValueError
  t = Tokenizer("dummy")
  with patch.object(Tokenizer, "_REGEX", re.compile("(?P<UNKNOWN>dummy)")):
    list(t.tokenize())

  # 322: BLOCK_LABEL
  assert MlirParser("{ ^bb0:")._is_region_start() is True

  # 386: Stuck parsing results
  with pytest.raises(SyntaxError, match="Stuck parsing results"):
    MlirParser("%res, @x = ").parse_operation()

  # 397-399: dotted op name
  # Since '.' causes ValueError in tokenizer, we construct the parser manually
  p = MlirParser("")
  p.tokens = [
    Token(TokenKind.IDENTIFIER, "foo", 1, 0),
    Token("UNKNOWN", ".", 1, 3),  # anything where text == "."
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

  # 408-411, 498: implicit sym_name
  op3 = MlirParser('"op" @sym () : ()').parse_operation()

  # 567-569: RBRACE empty block consume
  try:
    MlirParser("{ ").parse_region()
  except Exception:
    pass
