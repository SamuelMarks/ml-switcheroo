"""
SASS Tokenizer Definition.

Provides a Regex-based Lexer (`SassLexer`) that decomposes raw NVIDIA SASS assembly
strings into a stream of typed `Token` objects. It handles specific assembly
constructs like Predicates (`@P0`), Memory References (`c[0x0]`, `[R1]`),
Directives (`.headerflags`), and Modifiers (`.FTZ`).
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, Optional


class TokenType(Enum):
  """Enumeration of valid SASS token types."""

  # Structural
  LABEL_DEF = auto()  # L_1:
  DIRECTIVE = auto()  # .headerflags
  COMMENT = auto()  # // comment
  SEMICOLON = auto()  # ;
  COMMA = auto()  # ,

  # Operands
  PREDICATE = auto()  # @P0, @!P0
  REGISTER = auto()  # R0, -R1, |R2|
  MEMORY = auto()  # c[0x0][0x4], [R1], [R1 + 0x4]
  IMMEDIATE = auto()  # 0x1, 1.5, 42

  # Generic Identifiers (Opcodes, Label References)
  IDENTIFIER = auto()  # FADD, FADD.FTZ, L_1


@dataclass
class Token:
  """
  Represents a lexical unit.

  Attributes:
      kind: The type of token (e.g., OPCODE, REGISTER).
      value: The raw string value (e.g., "FADD", "R0").
      line: Line number in source (1-based).
      column: Column number in source (1-based).
  """

  kind: TokenType
  value: str
  line: int
  column: int


class SassLexer:
  """
  Regex-based Lexer for NVIDIA SASS assembly.
  """

  # Compiled Regex Patterns (Order matters for priority)
  PATTERNS = [
    # Comments: // ... or ; ... (if not end of instruction)
    # We strictly implement C-style // from Step 1 spec, but assume ; as EOI
    (TokenType.COMMENT, r"//.*"),
    # Structural
    (TokenType.SEMICOLON, r";"),
    (TokenType.COMMA, r","),
    # Structural Names
    (TokenType.LABEL_DEF, r"[a-zA-Z_][a-zA-Z0-9_\$\.]*:"),
    (TokenType.DIRECTIVE, r"\.[a-zA-Z_][a-zA-Z0-9_]*"),
    # Predicates: @P0, @!P0, @PT
    (TokenType.PREDICATE, r"@[!]?[a-zA-Z0-9_]+"),
    # Memory References:
    # Constant: c[0x0][0x0]
    # Global/Local: [R1], [R1 + 0x4]
    (TokenType.MEMORY, r"c\[[^\]]+\](\[[^\]]+\])?|\[[^\]]+\]"),
    # Registers: R0, -R0, |R0|, -|R0|, UKZ, SR_...
    # Must account for negation (-) and absolute (|) prefixes/wrappers
    # Using a broad pattern to catch modifiers attached to register names
    (TokenType.REGISTER, r"-?\|?[RU]?[RZS]\w*\|?"),
    # Immediates: Hex (0x...) or Numeric matching (Float/Int)
    # Note: Must come before Identifier to catch starting digits
    (TokenType.IMMEDIATE, r"[-+]?0x[0-9a-fA-F]+"),
    (TokenType.IMMEDIATE, r"[-+]?\d*\.\d+([eE][-+]?\d+)?"),
    (TokenType.IMMEDIATE, r"[-+]?\d+"),
    # Identifiers (Opcodes, Modifiers, Label refs): e.g. FADD.FTZ, L_1
    (TokenType.IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
  ]

  def __init__(self) -> None:
    """Initializes the lexer with compiled patterns."""
    self.regex_pairs = [(kind, re.compile(pattern)) for kind, pattern in self.PATTERNS]

  def tokenize(self, text: str) -> Generator[Token, None, None]:
    """
    Tokenizes the input string.

    Args:
        text: Raw SASS source code.

    Yields:
        Token objects.

    Raises:
        ValueError: If an unrecognized character sequence is encountered.
    """
    pos = 0
    line_num = 1
    line_start = 0
    length = len(text)

    while pos < length:
      # Handle Whitespace
      # We track newlines manually to update line_num
      match_ws = re.match(r"\s+", text[pos:])
      if match_ws:
        ws_str = match_ws.group(0)
        newlines = ws_str.count("\n")
        if newlines > 0:
          line_num += newlines
          # Find last newline to update line_start
          line_start = pos + ws_str.rfind("\n") + 1
        pos += len(ws_str)
        continue

      # Try Patterns
      match_found = False
      for kind, regex in self.regex_pairs:
        match = regex.match(text, pos)
        if match:
          val = match.group(0)
          column = pos - line_start + 1

          # Refinement: 'REGISTER' regex is broad (R\w*).
          # If it matched something like 'RET' or 'BRA' (Instructions),
          # we must reclassify as IDENTIFIER unless it's strictly a register pattern.
          # Basic register heuristics: R.., SR.., UR.., P.., UP.., UZ, RZ, PT
          if kind == TokenType.REGISTER:
            # Strip modifiers for check
            clean = val.replace("-", "").replace("|", "")
            if not (
              clean.startswith("R")
              or clean.startswith("SR")
              or clean.startswith("UR")
              or clean in ["RZ", "PT", "UP", "UZ"]
            ):
              # Fallback to Identifier (Opcode) if it doesn't look like a register
              # e.g. 'ROP' starts with R but is opcode?
              # SASS Registers are usually digit suffixes.
              if not any(c.isdigit() for c in clean) and clean not in ["RZ", "PT"]:
                kind = TokenType.IDENTIFIER

          yield Token(kind, val, line_num, column)

          pos += len(val)
          match_found = True
          break

      if not match_found:
        # Error context
        snippet = text[pos : min(pos + 10, length)]
        raise ValueError(f"Illegal character at line {line_num}, col {pos - line_start + 1}: '{snippet}...'")
