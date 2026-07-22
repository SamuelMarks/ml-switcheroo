"""MLIR Parser Regions."""

from typing import List, TYPE_CHECKING, Any
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol
from ml_switcheroo.core.mlir.nodes import BlockNode, RegionNode, TriviaNode, ModuleNode, ValueNode


if TYPE_CHECKING:
  from ml_switcheroo.core.mlir.parser_base import MlirParserBase
else:
  MlirParserBase = object


class MlirParserRegionsMixin(MlirParserBase):
  """Docstring."""

  if TYPE_CHECKING:

    def parse_type(self) -> Any:
      """Docstring."""
      ...

    def parse_operation(self) -> Any:
      """Docstring."""
      ...

  def _expect(self, kind: str) -> Any:
    """Consumes the token if it matches, else raises SyntaxError.  .



    Args:
        kind (str): The expected token kind or text.

    Returns:
        Token: The consumed token.

    Raises:
        SyntaxError: If the current token does not match the expectation.
    .
    """
    if not self.match(kind):
      cur = self.peek()
      raise SyntaxError(f"Expected {kind}, got {cur.kind} ('{cur.text}')")
    return self.consume()

  def _flush_trivia(self) -> List[TriviaNode]:
    """Returns and clears the accumulated trivia buffer.  .


    Returns:
        List[TriviaNode]: The collected whitespace and comments.
    .
    """
    t = self.trivia_buffer
    self.trivia_buffer = []
    return t

  def _absorb_trivia(self) -> None:
    """Consumes whitespace, comments, and newlines into the trivia buffer.  .

    This allows semantic parsing methods to ignore layout while preserving it.  .
    """
    while True:
      tk = self.peek()
      if tk.kind == TokenKind.EOF:
        break
      if tk.kind in (TokenKind.WHITESPACE, TokenKind.COMMENT, TokenKind.NEWLINE):
        self.consume()
        kmap = {"WHITESPACE": "whitespace", "NEWLINE": "newline", "COMMENT": "comment"}
        k_str = tk.kind.value if hasattr(tk.kind, "value") else str(tk.kind)
        self.trivia_buffer.append(TriviaNode(tk.text, kind=kmap.get(k_str, "whitespace")))
      else:
        break

  def parse(self) -> ModuleNode:
    """Top-level parsing entry point.  .


    Returns:
        ModuleNode: The root of the MLIR CST.
    .
    """
    return ModuleNode(body=self.parse_block(is_top_level=True))

  def parse_block(self, is_top_level: bool = False) -> BlockNode:
    """Parses a Basic Block.  .


    A block consists of an optional label (with arguments) and a list of operations.

    Args:
        is_top_level (bool): If True, treats the input as an implicit top-level module block
                             which may not have a label or braces.

    Returns:
        BlockNode: The parsed block structure.

    Raises:
        SyntaxError: If invalid tokens are encountered where an operation was expected.
    .
    """
    label = ""
    arguments = []
    self._absorb_trivia()
    leading = self._flush_trivia()

    if not is_top_level and self.match(TokenKind.BLOCK_LABEL):
      label = self.consume().text
      self._absorb_trivia()
      if self.match(Symbol.LPAREN):
        self.consume()
        while not self.match(Symbol.RPAREN):
          self._absorb_trivia()
          self._flush_trivia()  # Fix: Discard whitespace trivia inside arg list to prevent leaking

          if self.match(TokenKind.VAL_ID):  # pragma: no cover
            vn = self.consume().text
            self._absorb_trivia()
            self._flush_trivia()  # Fix

            if self.match(Symbol.COLON):  # pragma: no cover
              self.consume()
              self._absorb_trivia()
              t = self.parse_type()
              arguments.append((ValueNode(vn), t))
              self._absorb_trivia()
              if self.match(Symbol.COMMA):
                self.consume()
        self._expect(Symbol.RPAREN)
        self._absorb_trivia()
        if self.match(Symbol.COLON):
          self.consume()

    operations = []
    while True:
      self._absorb_trivia()
      pk = self.peek()
      if pk.kind in (TokenKind.EOF, TokenKind.BLOCK_LABEL) or pk.text == Symbol.RBRACE:
        break
      op = self.parse_operation()
      if op:
        operations.append(op)
      else:
        break
    return BlockNode(label=label, arguments=arguments, operations=operations, leading_trivia=leading)

  def parse_region(self) -> RegionNode:
    """Parses a region enclosed in braces."""
    blocks = []
    if self.match(Symbol.LBRACE):
      self.consume()

    self._absorb_trivia()

    # Create an implicit block if ops exist before any label
    if (
      self.peek().kind != TokenKind.BLOCK_LABEL
      and self.peek().kind != TokenKind.EOF
      and self.peek().text != Symbol.RBRACE
    ):
      implicit_block = self.parse_block(is_top_level=False)
      if implicit_block.operations:  # pragma: no cover
        blocks.append(implicit_block)

    while True:
      self._absorb_trivia()
      if self.peek().kind == TokenKind.EOF:
        break
      if self.match(TokenKind.BLOCK_LABEL):
        blocks.append(self.parse_block(is_top_level=False))
      else:
        break
      if self.match(Symbol.RBRACE):
        break  # pragma: no cover

    if self.match(Symbol.RBRACE):
      self.consume()
    return RegionNode(blocks=blocks)
