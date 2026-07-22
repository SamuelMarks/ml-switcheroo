"""MLIR Parser Ops."""

from typing import Optional, TYPE_CHECKING, Any
from ml_switcheroo.core.mlir.nodes import OperationNode, ValueNode, AttributeNode, TypeNode
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol

if TYPE_CHECKING:
  from ml_switcheroo.core.mlir.parser_base import MlirParserBase
else:
  MlirParserBase = object


class MlirParserOpsMixin(MlirParserBase):
  """Docstring."""

  if TYPE_CHECKING:
    pos: int

    def peek(self, offset: int = 0) -> Any:
      """Docstring."""
      ...

    def match(self, kind: str) -> bool:
      """Docstring."""
      ...

    def consume(self) -> Any:
      """Docstring."""
      ...

    def expect(self, kind: str) -> Any:
      """Docstring."""
      ...

    def _absorb_trivia(self) -> None:
      """Docstring."""
      ...

    def _flush_trivia(self) -> Any:
      """Docstring."""
      ...

    def _is_region_start(self) -> bool:
      """Docstring."""
      ...

    def parse_region(self) -> Any:
      """Docstring."""
      ...

  def parse_operation(self) -> Optional[OperationNode]:
    """Parses a single MLIR Operation.

    Structure:
    `%results = "op.name"(%operands) {attributes} ({regions}) : type`

    Returns:
        Optional[OperationNode]: The parsed operation, or None if no valid op start found.

    Raises:
        SyntaxError: If structural expectations (e.g. closing parens) are unmet.

    """
    results = []
    lh = 0
    eq_found = False
    while True:
      tk = self.peek(lh)
      if tk.kind in (TokenKind.EOF, TokenKind.NEWLINE, TokenKind.BLOCK_LABEL) or tk.text in [
        Symbol.LBRACE,
        Symbol.RBRACE,
      ]:
        break
      if tk.text == Symbol.EQUAL:
        eq_found = True
        break
      if lh > 20:
        break
      lh += 1

    if eq_found:
      while self.peek().text != Symbol.EQUAL:  # pragma: no cover
        start_pos = self.pos
        if self.match(TokenKind.VAL_ID):
          results.append(ValueNode(self.consume().text))
        elif self.match(Symbol.COMMA):
          self.consume()

        self._absorb_trivia()
        if self.peek().text == Symbol.EQUAL:
          break
        if self.pos == start_pos:
          raise SyntaxError(f"Stuck parsing results at {self.peek().text}")

      self.consume()

      self._absorb_trivia()
      self._flush_trivia()

    op_name = ""
    if self.match(TokenKind.STRING) or self.match(TokenKind.IDENTIFIER):
      op_name = self.consume().text
      while self.peek().text == ".":
        self.consume()
        if self.match(TokenKind.IDENTIFIER):  # pragma: no cover
          op_name += "." + self.consume().text
    else:
      return None

    self._absorb_trivia()
    name_trivia = self._flush_trivia()

    implicit_sym_name = None
    if self.match(TokenKind.SYM_ID):
      implicit_sym_name = self.consume().text
      if implicit_sym_name.startswith("@"):  # pragma: no cover
        implicit_sym_name = implicit_sym_name[1:]
      self._absorb_trivia()
      # don't flush trivia here to keep it attached to operands/attributes

    operands = []
    if self.peek().text == Symbol.LPAREN:
      self.consume()
      while not self.match(Symbol.RPAREN):
        self._absorb_trivia()
        if self.match(TokenKind.VAL_ID) or self.match(TokenKind.SYM_ID):
          operands.append(ValueNode(self.consume().text))
        elif self.match(Symbol.COMMA):
          self.consume()
        else:
          break
      self.expect(Symbol.RPAREN)

    self._absorb_trivia()
    self._flush_trivia()

    attributes = []
    if self.match(Symbol.LBRACE):
      if not self._is_region_start():
        self.consume()
        while not self.match(Symbol.RBRACE):
          self._absorb_trivia()
          if self.peek().kind == TokenKind.EOF:
            break
          if self.match(Symbol.RBRACE):
            break

          if self.match(TokenKind.IDENTIFIER) or self.match(TokenKind.STRING):  # pragma: no cover
            key = self.consume().text
            self._absorb_trivia()
            if self.match(Symbol.EQUAL):  # pragma: no cover
              self.consume()
              self._absorb_trivia()

              # ATTRIBUTE VALUE PARSING (Updated to handle nested brackets)
              val_s = []
              depth = 0
              while True:
                tk = self.peek()
                txt = tk.text

                # Delimiter check respecting nesting depth
                if depth == 0 and txt in [Symbol.COMMA, Symbol.RBRACE, Symbol.COLON]:
                  break

                if tk.kind == TokenKind.EOF:
                  break

                tk = self.consume()
                val_s.append(tk.text)

                # Check nesting
                if tk.text == Symbol.LBRACKET:
                  depth += 1
                elif tk.text == Symbol.RBRACKET:
                  depth -= 1

                if self.peek().kind == TokenKind.WHITESPACE:
                  self.consume()

              val_str = "".join(val_s).strip()

              tp = None
              if self.match(Symbol.COLON):
                self.consume()
                self._absorb_trivia()
                if self.match(TokenKind.TYPE):  # pragma: no cover
                  tp = self.consume().text
              attributes.append(AttributeNode(key, val_str, tp))
          self._absorb_trivia()
          if self.match(Symbol.COMMA):
            self.consume()
        self.expect(Symbol.RBRACE)

    self._absorb_trivia()
    self._flush_trivia()

    regions = []
    if self.peek().text == Symbol.LBRACE:
      if self._is_region_start():  # pragma: no cover
        regions.append(self.parse_region())

    if implicit_sym_name:
      # Prepend implicit sym_name to attributes so it's handled like standard MLIR {sym_name="x"}
      attributes.insert(0, AttributeNode("sym_name", f'"{implicit_sym_name}"'))

    self._absorb_trivia()
    res_types = []
    if self.match(Symbol.COLON):
      self._flush_trivia()
      self.consume()
      self._absorb_trivia()
      if self.match(Symbol.LPAREN):
        self.consume()
        while not self.match(Symbol.RPAREN):
          self._absorb_trivia()
          if self.match(TokenKind.TYPE) or self.match(TokenKind.REGION_TYPE):  # pragma: no cover
            self._flush_trivia()
            res_types.append(TypeNode(self.consume().text))
          if self.match(Symbol.COMMA):
            self.consume()
        self.consume()
      elif self.match(TokenKind.TYPE) or self.match(TokenKind.REGION_TYPE):  # pragma: no cover
        self._flush_trivia()
        res_types.append(TypeNode(self.consume().text))

    self._absorb_trivia()
    if self.match(TokenKind.ARROW):
      self.consume()
      self._absorb_trivia()
      if self.match(Symbol.LPAREN):
        self.consume()
        while not self.match(Symbol.RPAREN):
          self._absorb_trivia()
          if self.match(TokenKind.TYPE) or self.match(TokenKind.REGION_TYPE):  # pragma: no cover
            self._flush_trivia()
            res_types.append(TypeNode(self.consume().text))
          if self.match(Symbol.COMMA):
            self.consume()
        self.consume()
      elif self.match(TokenKind.TYPE) or self.match(TokenKind.REGION_TYPE):  # pragma: no cover
        self._flush_trivia()
        res_types.append(TypeNode(self.consume().text))

    self._absorb_trivia()
    trailing = self._flush_trivia()

    return OperationNode(
      op_name, results, operands, attributes, regions, res_types, name_trivia=name_trivia, trailing_trivia=trailing
    )
