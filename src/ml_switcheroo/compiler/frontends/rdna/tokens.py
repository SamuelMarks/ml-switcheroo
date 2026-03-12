"""
RDNA Tokenizer Definition.

Provides a Regex-based Lexer (`RdnaLexer`) that decomposes raw AMD RDNA assembly
strings into a stream of typed `Token` objects. It handles specific assembly
constructs like Registers (`s0`, `v[0:3]`), Immediates (`0xFF`),
Directives (`.text`), and Modifiers (`glc`).
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, Tuple


class TokenType(Enum):
  """Enumeration of valid RDNA token types."""

  # Structural
  LABEL_DEF = auto()  # L_1:
  DIRECTIVE = auto()  # .text
  COMMENT = auto()  # ; comment
  COMMA = auto()  # ,
  RBRACKET = auto()  # ] (used in range parsing)
  LBRACKET = auto()  # [ (used in range parsing)
  COLON = auto()  # : (used in range parsing)

  # Operands
  SGPR = auto()  # s0, s102
  VGPR = auto()  # v0, v255
  SPECIAL_REG = auto()  # exec, vcc, m0, scc
  IMMEDIATE = auto()  # 0x1, 1.5, 42
  MODIFIER = auto()  # glc, slc, off, vmcnt(0)

  # Generic Identifiers (Opcodes, Label References)
  IDENTIFIER = auto()  # v_add_f32


@dataclass
class Token:
  """
  Represents a lexical unit.

  Attributes:
      kind (TokenType): The type of token.
      value (str): The raw string value.
      line (int): Line number in source (1-based).
      column (int): Column number in source (1-based).
  """

  kind: TokenType
  value: str
  line: int
  column: int


class RdnaLexer:
  """
  Regex-based Lexer for AMD RDNA / GCN assembly.
  """

  # Compiled Regex Patterns (Order determines priority)
  # Note: RDNA uses ';' for comments
  PATTERNS: List[Tuple[TokenType, str]] = [
    # 1. Comments: ; ...
    (TokenType.COMMENT, r";.*"),
    # 2. Structural
    (TokenType.COMMA, r","),
    (TokenType.LBRACKET, r"\["),
    (TokenType.RBRACKET, r"\]"),
    (TokenType.COLON, r":"),
    # 3. Directives & Labels
    (TokenType.DIRECTIVE, r"\.[a-zA-Z_][a-zA-Z0-9_]*"),
    (TokenType.LABEL_DEF, r"[a-zA-Z_][a-zA-Z0-9_\$]*:"),
    # 4. Registers
    # SGPR: s followed by digits (s0, s10), typically lowercase
    (TokenType.SGPR, r"\bs[0-9]+\b"),
    # VGPR: v followed by digits (v0, v255)
    (TokenType.VGPR, r"\bv[0-9]+\b"),
    # Special Registers
    (TokenType.SPECIAL_REG, r"\b(exec|vcc|m0|scc|flat_scratch|xnack_mask)\b"),
    # 5. Modifiers
    # Handles simple modifiers (glc) and complex wait counters (vmcnt(0))
    # Removed trailing \b for parens match
    (TokenType.MODIFIER, r"(?:\b(glc|slc|dlc|off)\b|vmcnt\(\d+\)|lgkmcnt\(\d+\))"),
    # 6. Immediates
    # Hex
    (TokenType.IMMEDIATE, r"[-+]?0x[0-9a-fA-F]+"),
    # Float (scientific notation or decimal)
    (TokenType.IMMEDIATE, r"[-+]?\d*\.\d+([eE][-+]?\d+)?"),
    # Integer
    (TokenType.IMMEDIATE, r"[-+]?\d+"),
    # 7. Identifiers (Opcodes, Label Refs)
    # Allows dots (e.g. ds_read_b32)
    (TokenType.IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
  ]

  def __init__(self) -> None:
    """Initializes the lexer with compiled patterns."""
    self.regex_pairs = [(kind, re.compile(pattern)) for kind, pattern in self.PATTERNS]

  def tokenize(self, text: str) -> Generator[Token, None, None]:
    """
    Tokenizes the input string.

    Args:
        text (str): Raw RDNA source code.

    Yields:
        Token: Token objects.

    Raises:
        ValueError: If an unrecognized character sequence is encountered.
    """
    pos = 0
    line_num = 1
    line_start = 0
    length = len(text)

    while pos < length:
      # Handle Whitespace
      match_ws = re.match(r"\s+", text[pos:])
      if match_ws:
        ws_str = match_ws.group(0)
        newlines = ws_str.count("\n")
        if newlines > 0:
          line_num += newlines
          line_start = pos + ws_str.rfind("\n") + 1
        pos += len(ws_str)
        continue

      match_found = False
      for kind, regex in self.regex_pairs:
        match = regex.match(text, pos)
        if match:
          val = match.group(0)
          column = pos - line_start + 1

          # Special Case: LABEL_DEF regex matches the trailing colon
          yield Token(kind, val, line_num, column)

          pos += len(val)
          match_found = True
          break

      if not match_found:
        snippet = text[pos : min(pos + 10, length)]
        raise ValueError(f"Illegal character at line {line_num}, col {pos - line_start + 1}: '{snippet}...'")
