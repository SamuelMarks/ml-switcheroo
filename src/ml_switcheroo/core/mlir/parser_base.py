"""MLIR Parser Base."""

from typing import List
from ml_switcheroo.core.mlir.nodes import TriviaNode
from ml_switcheroo.core.mlir.lexer import Tokenizer, Token
from ml_switcheroo.core.mlir.tokens import TokenKind


class MlirParserBase:
  """Docstring."""

  def __init__(self, text: str):
    """Docstring."""
    self.tokenizer = Tokenizer(text)
    self.tokens = list(self.tokenizer.tokenize())
    self.pos = 0
    self.trivia_buffer: List[TriviaNode] = []

  def peek(self, offset: int = 0) -> Token:
    """Docstring."""
    idx = self.pos + offset
    if idx < len(self.tokens):
      return self.tokens[idx]
    return self.tokens[-1]

  def consume(self) -> Token:
    """Docstring."""
    tk = self.peek()
    if tk.kind != TokenKind.EOF:
      self.pos += 1
    return tk

  def match(self, kind: str) -> bool:
    """Docstring."""
    if self.peek().kind == kind or self.peek().text == kind:
      return True
    return False

  def expect(self, kind: str) -> Token:
    """Docstring."""
    if self.match(kind):
      return self.consume()
    raise SyntaxError(f"Expected {kind} but got {self.peek().kind} at line {self.peek().line}")

  def _flush_trivia(self) -> List[TriviaNode]:
    """Docstring."""
    res = list(self.trivia_buffer)
    self.trivia_buffer.clear()
    return res

  def _absorb_trivia(self) -> None:
    """Docstring."""
    while self.match(TokenKind.WHITESPACE) or self.match(TokenKind.NEWLINE) or self.match(TokenKind.COMMENT):
      tk = self.consume()
      if tk.kind == TokenKind.COMMENT:
        self.trivia_buffer.append(TriviaNode(content=tk.text))
