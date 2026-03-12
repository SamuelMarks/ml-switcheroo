"""
SASS Tokenizer Definition.

Provides a Regex-based Lexer (`SassLexer`) that decomposes raw NVIDIA SASS assembly
strings into a stream of typed `Token` objects.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, Tuple


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
      kind (TokenType): The type of token.
      value (str): The raw string content.
      line (int): Line number in source (1-based).
      column (int): Column number in source (1-based).
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
  PATTERNS: List[Tuple[TokenType, str]] = [
    (TokenType.COMMENT, r"//.*"),
    (TokenType.SEMICOLON, r";"),
    (TokenType.COMMA, r","),
    (TokenType.LABEL_DEF, r"[a-zA-Z_][a-zA-Z0-9_\$\.]*:"),
    (TokenType.DIRECTIVE, r"\.[a-zA-Z_][a-zA-Z0-9_]*"),
    (TokenType.PREDICATE, r"@[!]?[a-zA-Z0-9_]+"),
    # Constant: c[0x0][0x0], Global: [R1]
    (TokenType.MEMORY, r"c\[[^\]]+\](\[[^\]]+\])?|\[[^\]]+\]"),
    # Registers: R.., SR.., UR.., P.., UP.., UZ, RZ, PT (with optional negation/abs)
    (TokenType.REGISTER, r"-?\|?[RU]?[RZS]\w*\|?"),
    # Immediates
    (TokenType.IMMEDIATE, r"[-+]?0x[0-9a-fA-F]+"),
    (TokenType.IMMEDIATE, r"[-+]?\d*\.\d+([eE][-+]?\d+)?"),
    (TokenType.IMMEDIATE, r"[-+]?\d+"),
    # Identifiers
    (TokenType.IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
  ]

  def __init__(self) -> None:
    """Initializes the lexer with compiled patterns."""
    self.regex_pairs = [(kind, re.compile(pattern)) for kind, pattern in self.PATTERNS]

  def tokenize(self, text: str) -> Generator[Token, None, None]:
    """
    Tokenizes the input string.

    Args:
        text (str): Raw SASS source code.

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

          # Refinement: Differentiate Registers from Opcodes
          if kind == TokenType.REGISTER:
            clean = val.replace("-", "").replace("|", "")
            if not (
              clean.startswith("R")
              or clean.startswith("SR")
              or clean.startswith("UR")
              or clean in ["RZ", "PT", "UP", "UZ"]
            ):
              # Fallback if digits check logic fails for specific mnemonics
              if not any(c.isdigit() for c in clean) and clean not in [
                "RZ",
                "PT",
              ]:
                kind = TokenType.IDENTIFIER

          yield Token(kind, val, line_num, column)

          pos += len(val)
          match_found = True
          break

      if not match_found:
        snippet = text[pos : min(pos + 10, length)]
        raise ValueError(f"Illegal character at line {line_num}, col {pos - line_start + 1}: '{snippet}...'")
