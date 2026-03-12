from ml_switcheroo.core.mlir.parser import MlirParser, Token
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol


def test_parser_gap3():
  p = MlirParser("{")
  p.tokens = [
    Token(TokenKind.SYMBOL, "{", 1, 0),
    Token(TokenKind.MISMATCH, "dummy", 1, 1),
    Token(TokenKind.SYMBOL, "}", 1, 2),
    Token(TokenKind.EOF, "", 1, 3),
  ]
  p.pos = 0
  try:
    p.parse_region()
  except Exception:
    pass
