"""
RDNA Parser Implementation.

Parses a stream of `Token`s from the Lexer into `RdnaNode` ASTs.
"""

from typing import List, Optional, Union

from ml_switcheroo.compiler.frontends.rdna.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  LabelRef,
  Memory,
  Modifier,
  Operand,
  RdnaNode,
  SGPR,
  VGPR,
)
from ml_switcheroo.compiler.frontends.rdna.tokens import RdnaLexer, Token, TokenType


class RdnaParser:
  """
  Recursive descent parser for AMD RDNA / GCN assembly.
  """

  def __init__(self, code: str) -> None:
    """
    Initialize the parser.

    Args:
        code (str): The raw RDNA source string.
    """
    self.lexer = RdnaLexer()
    self.tokens = list(self.lexer.tokenize(code))
    self.pos = 0

  def parse(self) -> List[RdnaNode]:
    """
    Parses the entire code block.
    """
    nodes: List[RdnaNode] = []
    while not self._is_eof():
      node = self._parse_line()
      if node:
        nodes.append(node)
    return nodes

  # --- Recursive Descent Implementation ---

  def _peek(self, offset: int = 0) -> Optional[Token]:
    """Looks ahead at the pending token."""
    if self.pos + offset < len(self.tokens):
      return self.tokens[self.pos + offset]
    return None

  def _consume(self, kind: Optional[TokenType] = None) -> Token:
    """Consumes the current token."""
    token = self._peek()
    if not token:
      raise SyntaxError("Unexpected End of File.")

    if kind and token.kind != kind:
      raise SyntaxError(f"Expected {kind}, got {token.kind} ('{token.value}') at line {token.line}")

    self.pos += 1
    return token

  def _is_eof(self) -> bool:
    return self.pos >= len(self.tokens)

  def _match(self, kind: TokenType) -> bool:
    """Checks if current token matches kind."""
    token = self._peek()
    return token is not None and token.kind == kind

  def _parse_line(self) -> Optional[RdnaNode]:
    """Parses a top-level syntactic unit."""
    token = self._peek()
    if not token:
      return None

    if token.kind == TokenType.COMMENT:
      self._consume()
      raw = token.value
      clean = raw.lstrip(";").strip()
      return Comment(text=clean)

    if token.kind == TokenType.LABEL_DEF:
      self._consume()
      return Label(name=token.value[:-1])

    if token.kind == TokenType.DIRECTIVE:
      return self._parse_directive()

    if token.kind == TokenType.IDENTIFIER:
      return self._parse_instruction()

    bad_token = self._consume()
    raise SyntaxError(f"Unexpected token at line {bad_token.line}: {bad_token.value}")

  def _parse_directive(self) -> Directive:
    """Parses an assembler directive line."""
    tok = self._consume(TokenType.DIRECTIVE)
    name = tok.value[1:]

    params: List[str] = []
    while not self._is_eof():
      next_t = self._peek()
      if not next_t:
        break

      # Stop at start of next statement
      if next_t.kind in (
        TokenType.LABEL_DEF,
        TokenType.DIRECTIVE,
        TokenType.COMMENT,
      ):
        break

      if next_t.line > tok.line:
        break

      param_tok = self._consume()
      params.append(param_tok.value)

      if self._match(TokenType.COMMA):
        self._consume()

    return Directive(name=name, params=params)

  def _parse_instruction(self) -> Instruction:
    """
    Parses an RDNA instruction.
    Format: Opcode [Operands...]
    """
    op_tok = self._consume(TokenType.IDENTIFIER)
    opcode = op_tok.value
    operands: List[Operand] = []

    while not self._is_eof():
      peek = self._peek()
      if not peek or peek.line > op_tok.line or peek.kind == TokenType.COMMENT:
        break
      if peek.kind in (TokenType.LABEL_DEF, TokenType.DIRECTIVE):
        break

      operands.append(self._parse_operand())

      if self._match(TokenType.COMMA):
        self._consume()

    return Instruction(opcode=opcode, operands=operands)

  def _parse_operand(self) -> Operand:
    """Parses a single operand."""
    token = self._peek()
    if not token:
      raise SyntaxError("Unexpected EOF expecting operand")

    if token.kind == TokenType.SGPR or token.kind == TokenType.VGPR:
      reg_token = self._consume()
      reg_class = SGPR if token.kind == TokenType.SGPR else VGPR
      index = int(reg_token.value[1:])
      return reg_class(index=index, count=1)

    if token.kind == TokenType.IDENTIFIER:
      if token.value in ["s", "v"]:
        next_t = self._peek(1)
        if next_t and next_t.kind == TokenType.LBRACKET:
          return self._parse_register_range()

      val = token.value
      self._consume()
      return LabelRef(name=val)

    if token.kind == TokenType.MODIFIER:
      mod_tok = self._consume()
      return Modifier(name=mod_tok.value)

    if token.kind == TokenType.IMMEDIATE:
      self._consume()
      val_str = token.value
      is_hex = "0x" in val_str.lower()
      val = float(val_str) if "." in val_str and not is_hex else int(val_str, 16 if is_hex else 10)
      return Immediate(value=val, is_hex=is_hex)

    if token.kind == TokenType.SPECIAL_REG:
      tok = self._consume()
      return LabelRef(name=tok.value)

    raise SyntaxError(f"Unknown operand type: {token.kind} ({token.value})")

  def _parse_register_range(self) -> Union[SGPR, VGPR]:
    """Parses range syntax like s[0:3]."""
    base_tok = self._consume(TokenType.IDENTIFIER)
    reg_class = SGPR if base_tok.value == "s" else VGPR

    self._consume(TokenType.LBRACKET)
    start_tok = self._consume(TokenType.IMMEDIATE)
    self._consume(TokenType.COLON)
    end_tok = self._consume(TokenType.IMMEDIATE)
    self._consume(TokenType.RBRACKET)

    start_idx = int(start_tok.value)
    end_idx = int(end_tok.value)
    count = end_idx - start_idx + 1

    return reg_class(index=start_idx, count=count)
