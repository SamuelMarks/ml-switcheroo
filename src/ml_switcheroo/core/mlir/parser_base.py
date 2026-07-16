"""MLIR Parser Base."""

from typing import List  # pragma: no cover
from ml_switcheroo.core.mlir.nodes import TriviaNode  # pragma: no cover
from ml_switcheroo.core.mlir.lexer import Tokenizer, Token  # pragma: no cover
from ml_switcheroo.core.mlir.tokens import TokenKind  # pragma: no cover


# pragma: no cover
# pragma: no cover
class MlirParserBase:  # pragma: no cover
  """Docstring."""  # pragma: no cover

  # pragma: no cover
  def __init__(self, text: str):  # pragma: no cover
    """Docstring."""  # pragma: no cover
    self.tokenizer = Tokenizer(text)  # pragma: no cover
    self.tokens = list(self.tokenizer.tokenize())  # pragma: no cover
    self.pos = 0  # pragma: no cover
    self.trivia_buffer: List[TriviaNode] = []  # pragma: no cover

  # pragma: no cover
  def peek(self, offset: int = 0) -> Token:  # pragma: no cover
    """Docstring."""  # pragma: no cover
    idx = self.pos + offset  # pragma: no cover
    if idx < len(self.tokens):  # pragma: no cover
      return self.tokens[idx]  # pragma: no cover
    return self.tokens[-1]  # pragma: no cover

  # pragma: no cover
  def consume(self) -> Token:  # pragma: no cover
    """Docstring."""  # pragma: no cover
    tk = self.peek()  # pragma: no cover
    if tk.kind != TokenKind.EOF:  # pragma: no cover
      self.pos += 1  # pragma: no cover
    return tk  # pragma: no cover

  # pragma: no cover
  def match(self, kind: str) -> bool:  # pragma: no cover
    """Docstring."""  # pragma: no cover
    if self.peek().kind == kind or self.peek().text == kind:  # pragma: no cover
      return True  # pragma: no cover
    return False  # pragma: no cover

  # pragma: no cover
  def expect(self, kind: str) -> Token:  # pragma: no cover
    """Docstring."""  # pragma: no cover
    if self.match(kind):  # pragma: no cover
      return self.consume()  # pragma: no cover
    raise SyntaxError(f"Expected {kind} but got {self.peek().kind} at line {self.peek().line}")  # pragma: no cover

  # pragma: no cover
  def _flush_trivia(self) -> List[TriviaNode]:  # pragma: no cover
    """Docstring."""  # pragma: no cover
    res = list(self.trivia_buffer)  # pragma: no cover
    self.trivia_buffer.clear()  # pragma: no cover
    return res  # pragma: no cover

  # pragma: no cover
  def _absorb_trivia(self) -> None:  # pragma: no cover
    """Docstring."""  # pragma: no cover
    while (
      self.match(TokenKind.WHITESPACE) or self.match(TokenKind.NEWLINE) or self.match(TokenKind.COMMENT)
    ):  # pragma: no cover
      tk = self.consume()  # pragma: no cover
      if tk.kind == TokenKind.COMMENT:  # pragma: no cover
        self.trivia_buffer.append(TriviaNode(content=tk.text))  # pragma: no cover
