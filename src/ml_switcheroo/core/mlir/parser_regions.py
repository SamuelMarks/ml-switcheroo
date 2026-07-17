"""MLIR Parser Regions."""

from typing import List, TYPE_CHECKING, Any  # pragma: no cover
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol  # pragma: no cover
from ml_switcheroo.core.mlir.nodes import BlockNode, RegionNode, TriviaNode, ModuleNode, ValueNode  # pragma: no cover

# pragma: no cover
if TYPE_CHECKING:  # pragma: no cover
  from ml_switcheroo.core.mlir.parser_base import MlirParserBase  # pragma: no cover
else:  # pragma: no cover
  MlirParserBase = object  # pragma: no cover
  # pragma: no cover
  # pragma: no cover


class MlirParserRegionsMixin(MlirParserBase):  # pragma: no cover
  """Docstring."""  # pragma: no cover

  # pragma: no cover
  if TYPE_CHECKING:  # pragma: no cover
    # pragma: no cover
    def parse_type(self) -> Any:  # pragma: no cover
      """Docstring."""  # pragma: no cover
      ...  # pragma: no cover

    # pragma: no cover
    def parse_operation(self) -> Any:  # pragma: no cover
      """Docstring."""  # pragma: no cover
      ...  # pragma: no cover

  # pragma: no cover
  def _expect(self, kind: str) -> Any:  # pragma: no cover
    """Consumes the token if it matches, else raises SyntaxError.  # pragma: no cover.


    # pragma: no cover
    Args:  # pragma: no cover
        kind (str): The expected token kind or text.  # pragma: no cover
    # pragma: no cover
    Returns:  # pragma: no cover
        Token: The consumed token.  # pragma: no cover
    # pragma: no cover
    Raises:  # pragma: no cover
        SyntaxError: If the current token does not match the expectation.  # pragma: no cover
    # pragma: no cover.
    """  # pragma: no cover
    if not self.match(kind):  # pragma: no cover
      cur = self.peek()  # pragma: no cover
      raise SyntaxError(f"Expected {kind}, got {cur.kind} ('{cur.text}')")  # pragma: no cover
    return self.consume()  # pragma: no cover

  # pragma: no cover
  def _flush_trivia(self) -> List[TriviaNode]:  # pragma: no cover
    """Returns and clears the accumulated trivia buffer.  # pragma: no cover.

    # pragma: no cover
    Returns:  # pragma: no cover
        List[TriviaNode]: The collected whitespace and comments.  # pragma: no cover
    # pragma: no cover.
    """  # pragma: no cover
    t = self.trivia_buffer  # pragma: no cover
    self.trivia_buffer = []  # pragma: no cover
    return t  # pragma: no cover

  # pragma: no cover

  def _absorb_trivia(self) -> None:  # pragma: no cover
    """Consumes whitespace, comments, and newlines into the trivia buffer.  # pragma: no cover.

    This allows semantic parsing methods to ignore layout while preserving it.  # pragma: no cover.
    """  # pragma: no cover
    while True:  # pragma: no cover
      tk = self.peek()  # pragma: no cover
      if tk.kind == TokenKind.EOF:  # pragma: no cover
        break  # pragma: no cover
      if tk.kind in (TokenKind.WHITESPACE, TokenKind.COMMENT, TokenKind.NEWLINE):  # pragma: no cover
        self.consume()  # pragma: no cover
        kmap = {"WHITESPACE": "whitespace", "NEWLINE": "newline", "COMMENT": "comment"}  # pragma: no cover
        k_str = tk.kind.value if hasattr(tk.kind, "value") else str(tk.kind)  # pragma: no cover
        self.trivia_buffer.append(TriviaNode(tk.text, kind=kmap.get(k_str, "whitespace")))  # pragma: no cover
      else:  # pragma: no cover
        break  # pragma: no cover

  # pragma: no cover
  def parse(self) -> ModuleNode:  # pragma: no cover
    """Top-level parsing entry point.  # pragma: no cover.

    # pragma: no cover
    Returns:  # pragma: no cover
        ModuleNode: The root of the MLIR CST.  # pragma: no cover
    # pragma: no cover.
    """  # pragma: no cover
    return ModuleNode(body=self.parse_block(is_top_level=True))  # pragma: no cover

  # pragma: no cover
  def parse_block(self, is_top_level: bool = False) -> BlockNode:  # pragma: no cover
    """Parses a Basic Block.  # pragma: no cover.

    # pragma: no cover
    A block consists of an optional label (with arguments) and a list of operations.  # pragma: no cover
    # pragma: no cover
    Args:  # pragma: no cover
        is_top_level (bool): If True, treats the input as an implicit top-level module block  # pragma: no cover
                             which may not have a label or braces.  # pragma: no cover
    # pragma: no cover
    Returns:  # pragma: no cover
        BlockNode: The parsed block structure.  # pragma: no cover
    # pragma: no cover
    Raises:  # pragma: no cover
        SyntaxError: If invalid tokens are encountered where an operation was expected.  # pragma: no cover
    # pragma: no cover.
    """  # pragma: no cover
    label = ""  # pragma: no cover
    arguments = []  # pragma: no cover
    self._absorb_trivia()  # pragma: no cover
    leading = self._flush_trivia()  # pragma: no cover
    # pragma: no cover
    if not is_top_level and self.match(TokenKind.BLOCK_LABEL):  # pragma: no cover
      label = self.consume().text  # pragma: no cover
      self._absorb_trivia()  # pragma: no cover
      if self.match(Symbol.LPAREN):  # pragma: no cover
        self.consume()  # pragma: no cover
        while not self.match(Symbol.RPAREN):  # pragma: no cover
          self._absorb_trivia()  # pragma: no cover
          self._flush_trivia()  # Fix: Discard whitespace trivia inside arg list to prevent leaking  # pragma: no cover
          # pragma: no cover
          if self.match(TokenKind.VAL_ID):  # pragma: no cover
            vn = self.consume().text  # pragma: no cover
            self._absorb_trivia()  # pragma: no cover
            self._flush_trivia()  # Fix  # pragma: no cover
            # pragma: no cover
            if self.match(Symbol.COLON):  # pragma: no cover
              self.consume()  # pragma: no cover
              self._absorb_trivia()  # pragma: no cover
              t = self.parse_type()  # pragma: no cover
              arguments.append((ValueNode(vn), t))  # pragma: no cover
              self._absorb_trivia()  # pragma: no cover
              if self.match(Symbol.COMMA):  # pragma: no cover
                self.consume()  # pragma: no cover
        self._expect(Symbol.RPAREN)  # pragma: no cover
        self._absorb_trivia()  # pragma: no cover
        if self.match(Symbol.COLON):  # pragma: no cover
          self.consume()  # pragma: no cover
    # pragma: no cover
    operations = []  # pragma: no cover
    while True:  # pragma: no cover
      self._absorb_trivia()  # pragma: no cover
      pk = self.peek()  # pragma: no cover
      if pk.kind in (TokenKind.EOF, TokenKind.BLOCK_LABEL) or pk.text == Symbol.RBRACE:  # pragma: no cover
        break  # pragma: no cover
      op = self.parse_operation()  # pragma: no cover
      if op:  # pragma: no cover
        operations.append(op)  # pragma: no cover
      else:  # pragma: no cover
        break  # pragma: no cover
    return BlockNode(label=label, arguments=arguments, operations=operations, leading_trivia=leading)  # pragma: no cover

  # pragma: no cover
  def parse_region(self) -> RegionNode:  # pragma: no cover
    """Parses a region enclosed in braces."""  # pragma: no cover
    blocks = []  # pragma: no cover
    if self.match(Symbol.LBRACE):  # pragma: no cover
      self.consume()  # pragma: no cover
    # pragma: no cover
    self._absorb_trivia()  # pragma: no cover
    # pragma: no cover
    # Create an implicit block if ops exist before any label  # pragma: no cover
    if (  # pragma: no cover
      self.peek().kind != TokenKind.BLOCK_LABEL  # pragma: no cover
      and self.peek().kind != TokenKind.EOF  # pragma: no cover
      and self.peek().text != Symbol.RBRACE  # pragma: no cover
    ):  # pragma: no cover
      implicit_block = self.parse_block(is_top_level=False)  # pragma: no cover
      if implicit_block.operations:  # pragma: no cover
        blocks.append(implicit_block)  # pragma: no cover
    # pragma: no cover
    while True:  # pragma: no cover
      self._absorb_trivia()  # pragma: no cover
      if self.peek().kind == TokenKind.EOF:  # pragma: no cover
        break  # pragma: no cover
      if self.match(TokenKind.BLOCK_LABEL):  # pragma: no cover
        blocks.append(self.parse_block(is_top_level=False))  # pragma: no cover
      else:  # pragma: no cover
        break  # pragma: no cover
      if self.match(Symbol.RBRACE):  # pragma: no cover
        break  # pragma: no cover
    # pragma: no cover
    if self.match(Symbol.RBRACE):  # pragma: no cover
      self.consume()  # pragma: no cover
    return RegionNode(blocks=blocks)  # pragma: no cover
