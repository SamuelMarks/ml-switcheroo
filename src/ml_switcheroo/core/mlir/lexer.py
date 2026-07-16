"""MLIR Lexer."""

import re
from dataclasses import dataclass
from typing import Generator
from ml_switcheroo.core.mlir.tokens import TokenKind


@dataclass
class Token:
  """Represents a lexical token extracted from the source string."""

  kind: str
  text: str
  line: int
  col: int


class Tokenizer:
  """Lexical analyzer for MLIR syntax."""

  PATTERN_DEFS = [
    (TokenKind.COMMENT, r"//[^\n]*"),
    (TokenKind.STRING, r'"(?:[^"\\]|\\.)*"'),
    (TokenKind.REGION_TYPE, r"!sw\.type<[^>]+>"),
    (TokenKind.TYPE, r"![a-zA-Z_0-9\.<>]+|tensor<[^>]+>|[iuf]\d+|index|none"),
    (TokenKind.VAL_ID, r"%[a-zA-Z_0-9]+|%\d+"),
    (TokenKind.SYM_ID, r"@[a-zA-Z_0-9]+"),
    (TokenKind.BLOCK_LABEL, r"\^[a-zA-Z_0-9]+"),
    (TokenKind.ARROW, r"->"),
    (TokenKind.SYMBOL, r"[(){}\[\],:=]"),
    (TokenKind.IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_$.]*"),
    (TokenKind.NUMBER, r"-?\d+(?:\.\d+)?"),
    (TokenKind.NEWLINE, r"\n"),
    (TokenKind.WHITESPACE, r"[ \t]+"),
    (TokenKind.MISMATCH, r"."),
  ]

  _REGEX = re.compile("|".join(f"(?P<{kind.value}>{pattern})" for kind, pattern in PATTERN_DEFS))

  def __init__(self, text: str):
    """Initializes the tokenizer."""
    self.text = text

  def tokenize(self) -> Generator[Token, None, None]:
    """Yields tokens from the source text one by one."""
    line_num = 1
    line_start = 0
    for mo in self._REGEX.finditer(self.text):
      kind_str = mo.lastgroup
      value = mo.group()
      col = mo.start() - line_start

      try:
        kind = TokenKind(kind_str)
      except ValueError:
        kind = kind_str  # type: ignore

      if kind == TokenKind.NEWLINE:
        yield Token(kind, value, line_num, col)
        line_num += 1
        line_start = mo.end()
      elif kind == TokenKind.MISMATCH:
        raise ValueError(f"Unexpected character {value!r} on line {line_num}:{col}")
      else:
        yield Token(kind, value, line_num, col)
    yield Token(TokenKind.EOF, "", line_num, 0)
