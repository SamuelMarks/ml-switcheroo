"""
SASS Parser Implementation.

This module provides the `SassParser`, a recursive descent parser that converts
a stream of tokens (from `SassLexer`) into a Structural AST defined in `nodes.py`.

Capabilities:
- Parses Instructions, Labels, Directives, and Comments.
- Handles complex Operands (Memory, Predicates, Negated Registers).
- Supports parsing of basic blocks markers.
- Ignores potential stray semicolons.
"""

from typing import List, Optional, Union, Any

from ml_switcheroo.core.sass.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  Memory,
  Operand,
  Predicate,
  Register,
  SassNode,
)
from ml_switcheroo.core.sass.tokens import SassLexer, Token, TokenType

# Extended Operand Type for Label References
try:
  from dataclasses import dataclass

  @dataclass
  class LabelRef(Operand):
    name: str

    def __str__(self) -> str:
      return self.name
except ImportError:
  pass


class SassParser:
  """
  Recursive descent parser for NVIDIA SASS.
  """

  def __init__(self, code: str):
    """
    Initialize the parser.

    Args:
        code: The raw SASS source string.
    """
    self.lexer = SassLexer()
    self.tokens = list(self.lexer.tokenize(code))
    self.pos = 0

  def parse(self) -> List[SassNode]:
    """
    Parses the entire code block.

    Returns:
        A list of AST nodes.
    """
    nodes = []
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
    """
    Consumes the current token.

    Args:
        kind: If provided, enforces that the current token matches this type.

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
    return self.pos >= len(self.tokens)

  def _match(self, kind: TokenType) -> bool:
    """Checks if current token matches kind."""
    token = self._peek()
    return token is not None and token.kind == kind

  def _parse_line(self) -> Optional[SassNode]:
    """
    Parses a top-level syntactic unit.
    """
    token = self._peek()
    if not token:
      return None

    # Handle potentially loose semicolons explicitly to avoid errors
    if token.kind == TokenType.SEMICOLON:
      self._consume()
      return None  # Skip empty statement

    # 1. Comment
    if token.kind == TokenType.COMMENT:
      self._consume()
      raw = token.value
      clean = raw.lstrip("/;").strip()
      return Comment(text=clean)

    # 2. Label definition (e.g. L_1:)
    if token.kind == TokenType.LABEL_DEF:
      self._consume()
      return Label(name=token.value[:-1])

    # 3. Directive (e.g. .headerflags)
    if token.kind == TokenType.DIRECTIVE:
      return self._parse_directive()

    # 4. Instruction
    if token.kind == TokenType.PREDICATE or token.kind == TokenType.IDENTIFIER:
      return self._parse_instruction()

    # Unhandled token error
    bad_token = self._consume()
    raise SyntaxError(f"Unexpected token at line {bad_token.line}: {bad_token.value}")

  def _parse_directive(self) -> Directive:
    """Parses an assembler directive line."""
    tok = self._consume(TokenType.DIRECTIVE)
    name = tok.value[1:]  # Strip leading dot

    params = []

    # Consume generic args until structure boundary
    while not self._is_eof():
      next_t = self._peek()
      if not next_t:
        break

      if next_t.kind in (TokenType.LABEL_DEF, TokenType.DIRECTIVE, TokenType.COMMENT, TokenType.SEMICOLON):
        break

      # Safety check on newlines if tokens carry that info (usually handled by whitespace regex)
      if next_t.line > tok.line:
        break

      param_tok = self._consume()
      params.append(param_tok.value)

      if self._match(TokenType.COMMA):
        self._consume()

    # Handle optional semi-colon terminator for directives
    if self._match(TokenType.SEMICOLON):
      self._consume()

    return Directive(name=name, params=params)

  def _parse_instruction(self) -> Instruction:
    """
    Parses a SASS instruction.
    Format: [Predicate] Opcode [Operands...] [;]
    """
    # 1. Optional Predicate
    predicate = None
    if self._match(TokenType.PREDICATE):
      pred_tok = self._consume()
      raw_pred = pred_tok.value
      if raw_pred.startswith("@!"):
        predicate = Predicate(name=raw_pred[2:], negated=True)
      else:
        predicate = Predicate(name=raw_pred[1:], negated=False)

    # 2. Opcode
    op_tok = self._consume(TokenType.IDENTIFIER)
    opcode = op_tok.value

    # 3. Operands (Comma Separated)
    operands = []

    while not self._is_eof():
      if self._match(TokenType.SEMICOLON):
        self._consume()
        break

      peek = self._peek()
      if not peek or peek.line > op_tok.line or peek.kind == TokenType.COMMENT:
        break

      operands.append(self._parse_operand())

      if self._match(TokenType.COMMA):
        self._consume()
      else:
        pass

    return Instruction(opcode=opcode, operands=operands, predicate=predicate)

  def _parse_operand(self) -> Operand:
    """Parses a single operand."""
    token = self._peek()
    if not token:
      raise SyntaxError("Unexpected EOF expecting operand")

    if token.kind == TokenType.REGISTER:
      self._consume()
      return self._parse_register_str(token.value)

    if token.kind == TokenType.MEMORY:
      self._consume()
      return self._parse_memory_str(token.value)

    if token.kind == TokenType.IMMEDIATE:
      self._consume()
      val_str = token.value
      is_hex = "0x" in val_str.lower()
      val = float(val_str) if "." in val_str and not is_hex else int(val_str, 16 if is_hex else 10)
      return Immediate(value=val, is_hex=is_hex)

    if token.kind == TokenType.PREDICATE:
      self._consume()
      raw = token.value
      return Predicate(name=raw.lstrip("@!"), negated="!" in raw)

    if token.kind == TokenType.IDENTIFIER:
      self._consume()
      return LabelRef(name=token.value)

    raise SyntaxError(f"Unknown operand type: {token.kind} ({token.value})")

  def _parse_register_str(self, raw: str) -> Register:
    """Parses register attributes."""
    negated = raw.startswith("-")
    clean = raw.lstrip("-")
    absolute = clean.startswith("|") and clean.endswith("|")
    name = clean.strip("|")
    return Register(name=name, negated=negated, absolute=absolute)

  def _parse_memory_str(self, raw: str) -> Memory:
    """Parses memory string c[...] or [...]."""
    if raw.startswith("c["):
      inner = raw[1:]
      import re

      matches = re.findall(r"\[(.*?)\]", inner)
      if len(matches) == 2:
        bank, offset_str = matches
        base_str = f"c[{bank}]"
        offset = int(offset_str, 16)
        return Memory(base=base_str, offset=offset)
      else:
        bank = matches[0] if matches else "0x0"
        return Memory(base=f"c[{bank}]", offset=0)

    inner = raw.strip("[]")
    if "+" in inner:
      parts = inner.split("+")
      base_reg = parts[0].strip()
      off_str = parts[1].strip()
      offset = int(off_str, 16) if "0x" in off_str else int(off_str)
      return Memory(base=Register(name=base_reg), offset=offset)

    return Memory(base=Register(name=inner))
