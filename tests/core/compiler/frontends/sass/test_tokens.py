"""Test suite for the Tokens module."""

import pytest
from ml_switcheroo.core.compiler.frontends.sass.tokens import SassLexer, TokenType


def test_lexer_empty():
  """Verifies the behavior of lexer empty."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize(""))
  assert len(tokens) == 0


def test_lexer_whitespace():
  """Verifies the behavior of lexer whitespace."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("   \n  \t "))
  assert len(tokens) == 0


def test_lexer_comment():
  """Verifies the behavior of lexer comment."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("// hello"))
  assert len(tokens) == 1
  assert tokens[0].kind == TokenType.COMMENT
  assert tokens[0].value == "// hello"


def test_lexer_semicolon_comma():
  """Verifies the behavior of lexer semicolon comma."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("; ,"))
  assert len(tokens) == 2
  assert tokens[0].kind == TokenType.SEMICOLON
  assert tokens[1].kind == TokenType.COMMA


def test_lexer_label():
  """Verifies the behavior of lexer label."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("L_1:"))
  assert len(tokens) == 1
  assert tokens[0].kind == TokenType.LABEL_DEF
  assert tokens[0].value == "L_1:"


def test_lexer_directive():
  """Verifies the behavior of lexer directive."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize(".headerflags"))
  assert len(tokens) == 1
  assert tokens[0].kind == TokenType.DIRECTIVE
  assert tokens[0].value == ".headerflags"


def test_lexer_predicate():
  """Verifies the behavior of lexer predicate."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("@P0 @!P1"))
  assert len(tokens) == 2
  assert tokens[0].kind == TokenType.PREDICATE
  assert tokens[0].value == "@P0"
  assert tokens[1].kind == TokenType.PREDICATE
  assert tokens[1].value == "@!P1"


def test_lexer_register():
  """Verifies the behavior of lexer register."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("R0 -R1 |R2| RZ PT"))
  assert len(tokens) == 5
  assert tokens[0].kind == TokenType.REGISTER
  assert tokens[1].kind == TokenType.REGISTER
  assert tokens[2].kind == TokenType.REGISTER
  assert tokens[3].kind == TokenType.REGISTER
  assert tokens[4].kind in [TokenType.REGISTER, TokenType.IDENTIFIER]


def test_lexer_identifier_fallback():
  """Verifies the behavior of lexer identifier fallback."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("SADD"))
  assert len(tokens) == 1
  assert tokens[0].kind == TokenType.IDENTIFIER


def test_lexer_memory():
  """Verifies the behavior of lexer memory."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("c[0x0][0x4] [R1]"))
  assert len(tokens) == 2
  assert tokens[0].kind == TokenType.MEMORY
  assert tokens[1].kind == TokenType.MEMORY


def test_lexer_immediate():
  """Verifies the behavior of lexer immediate."""
  lexer = SassLexer()
  tokens = list(lexer.tokenize("0x1 1.5 42 -10"))
  assert len(tokens) == 4
  for t in tokens:
    assert t.kind == TokenType.IMMEDIATE


def test_lexer_invalid():
  """Verifies the behavior of lexer invalid."""
  lexer = SassLexer()
  with pytest.raises(ValueError, match="Illegal character"):
    list(lexer.tokenize("?"))
