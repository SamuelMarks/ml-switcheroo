"""
RDNA Parser Implementation.

This module provides the `RdnaParser`, a recursive descent parser that converts
a stream of tokens (from `RdnaLexer`) into a Structural AST defined in `nodes.py`.

Capabilities:
- Parses Instructions, Labels, Directives, and Comments.
- Handles complex Operands (Memory, Registers, Immediates, Modifiers).
- Supports parsing of register ranges (e.g. `v[0:3]`).
"""

from typing import List, Optional, Union

from ml_switcheroo.core.rdna.nodes import (
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
from ml_switcheroo.core.rdna.tokens import RdnaLexer, Token, TokenType


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

    Returns:
        List[RdnaNode]: A list of AST nodes.
    """
    nodes: List[RdnaNode] = []
    while not self._is_eof():
      node = self._parse_line()
      if node:
        nodes.append(node)
    return nodes

  # --- Recursive Descent Implementation ---

  def _peek(self, offset: int = 0) -> Optional[Token]:
    """
    Looks ahead at the pending token.

    Args:
        offset (int): How many tokens to look ahead.

    Returns:
        Optional[Token]: The token at current pos + offset, or None if EOF.
    """
    if self.pos + offset < len(self.tokens):
      return self.tokens[self.pos + offset]
    return None

  def _consume(self, kind: Optional[TokenType] = None) -> Token:
    """
    Consumes the current token.

    Args:
        kind (Optional[TokenType]): If provided, enforces that the current token matches this type.

    Returns:
        Token: The consumed token.

    Raises:
        SyntaxError: If end of file or type mismatch.
    """
    token = self._peek()
    if not token:
      raise SyntaxError("Unexpected End of File.")

    if kind and token.kind != kind:
      raise SyntaxError(f"Expected {kind}, got {token.kind} ('{token.value}') at line {token.line}")

    self.pos += 1
    return token

  def _is_eof(self) -> bool:
    """Returns True if no more tokens are available."""
    return self.pos >= len(self.tokens)

  def _match(self, kind: TokenType) -> bool:
    """Checks if current token matches kind."""
    token = self._peek()
    return token is not None and token.kind == kind

  def _parse_line(self) -> Optional[RdnaNode]:
    """
    Parses a top-level syntactic unit.

    Returns:
        Optional[RdnaNode]: The parsed node, or None if empty line/format quirk.
    """
    token = self._peek()
    if not token:
      return None

    # 1. Comment
    if token.kind == TokenType.COMMENT:
      self._consume()
      raw = token.value
      clean = raw.lstrip(";").strip()
      return Comment(text=clean)

    # 2. Label definition (e.g. L_1:)
    if token.kind == TokenType.LABEL_DEF:
      self._consume()
      # Strip trailing colon
      return Label(name=token.value[:-1])

    # 3. Directive (e.g. .text)
    if token.kind == TokenType.DIRECTIVE:
      return self._parse_directive()

    # 4. Instruction (Starts with Identifier/Opcode)
    if token.kind == TokenType.IDENTIFIER:
      return self._parse_instruction()

    # Unhandled token error
    bad_token = self._consume()
    raise SyntaxError(f"Unexpected token at line {bad_token.line}: {bad_token.value}")

  def _parse_directive(self) -> Directive:
    """Parses an assembler directive line."""
    tok = self._consume(TokenType.DIRECTIVE)
    name = tok.value[1:]  # Strip leading dot

    params: List[str] = []

    # Consume generic args until structure boundary (newline implication)
    # Since lexer doesn't explicitly yield NEWLINE tokens in the stream,
    # we rely on logic that directives are one-liners and stop at next
    # label/directive/comment or vastly different structure.
    while not self._is_eof():
      next_t = self._peek()
      if not next_t:
        break

      # Stop if we hit start of new statement
      if next_t.kind in (
        TokenType.LABEL_DEF,
        TokenType.DIRECTIVE,
        TokenType.COMMENT,
      ):
        break

      # Heuristic: Check line numbers if tokens carry them
      if next_t.line > tok.line:
        break

      param_tok = self._consume()
      # Try to keep basic structure if it's a structural token?
      # Or just append value string.
      params.append(param_tok.value)

      if self._match(TokenType.COMMA):
        self._consume()

    return Directive(name=name, params=params)

  def _parse_instruction(self) -> Instruction:
    """
    Parses an RDNA instruction.
    Format: Opcode [Operands...]
    """
    # 1. Opcode
    op_tok = self._consume(TokenType.IDENTIFIER)
    opcode = op_tok.value

    # 2. Operands
    operands: List[Operand] = []

    while not self._is_eof():
      peek = self._peek()
      # Stop if newline or next structural element
      if not peek or peek.line > op_tok.line or peek.kind == TokenType.COMMENT:
        break
      if peek.kind in (TokenType.LABEL_DEF, TokenType.DIRECTIVE):
        break

      operands.append(self._parse_operand())

      # Optional comma separator
      if self._match(TokenType.COMMA):
        self._consume()

    return Instruction(opcode=opcode, operands=operands)

  def _parse_operand(self) -> Operand:
    """Parses a single operand."""
    token = self._peek()
    if not token:
      raise SyntaxError("Unexpected EOF expecting operand")

    # 1. SGPR / VGPR (Check for Range syntax [0:3])
    if token.kind == TokenType.SGPR or token.kind == TokenType.VGPR:
      reg_token = self._consume()
      reg_class = SGPR if token.kind == TokenType.SGPR else VGPR
      prefix = "s" if token.kind == TokenType.SGPR else "v"

      index = int(reg_token.value[1:])
      return reg_class(index=index, count=1)

    # 2. Identifier (Could be Opcode, LabelRef, Modifier, or Register base for range)
    if token.kind == TokenType.IDENTIFIER:
      # Check for register semantics (s or v followed by bracket)
      if token.value in ["s", "v"]:
        next_t = self._peek(1)
        if next_t and next_t.kind == TokenType.LBRACKET:
          return self._parse_register_range()

      # Label Ref or Modifier
      val = token.value
      self._consume()
      # If it's a modifier keyword but lexed as identifier (if strict mode off)
      # But we have MODIFIER token type.
      return LabelRef(name=val)

    # 3. Modifier (glc, slc, off)
    if token.kind == TokenType.MODIFIER:
      mod_tok = self._consume()
      return Modifier(name=mod_tok.value)

    # 4. Immediate
    if token.kind == TokenType.IMMEDIATE:
      self._consume()
      val_str = token.value
      is_hex = "0x" in val_str.lower()
      val = float(val_str) if "." in val_str and not is_hex else int(val_str, 16 if is_hex else 10)
      return Immediate(value=val, is_hex=is_hex)

    # 5. Specialized Registers
    if token.kind == TokenType.SPECIAL_REG:
      # Treated as Named LabelRef or specialized Operand?
      # Let's map to Modifier or create explicit if logic requires.
      # For synthesis, just name string works.
      tok = self._consume()
      return LabelRef(name=tok.value)

    raise SyntaxError(f"Unknown operand type: {token.kind} ({token.value})")

  def _parse_register_range(self) -> Union[SGPR, VGPR]:
    """
    Parses range syntax like s[0:3].
    Assumes current peek is the base identifier 's' or 'v'.
    """
    base_tok = self._consume(TokenType.IDENTIFIER)
    # Determine class
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
