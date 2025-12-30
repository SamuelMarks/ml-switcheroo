"""
MLIR Token Definitions.

Defines the enumerations for Token Kinds and Symbols used by the Lexer and Parser.
"""

from enum import Enum


class TokenKind(str, Enum):
  """Enumeration of Lexer Token Types."""

  COMMENT = "COMMENT"
  STRING = "STRING"
  REGION_TYPE = "REGION_TYPE"
  TYPE = "TYPE"
  VAL_ID = "VAL_ID"
  SYM_ID = "SYM_ID"
  BLOCK_LABEL = "BLOCK_LABEL"
  ARROW = "ARROW"
  SYMBOL = "SYMBOL"
  IDENTIFIER = "IDENTIFIER"
  NUMBER = "NUMBER"
  NEWLINE = "NEWLINE"
  WHITESPACE = "WHITESPACE"
  MISMATCH = "MISMATCH"
  EOF = "EOF"


class Symbol(str, Enum):
  """Enumeration of Punctuation Symbols."""

  LBRACE = "{"
  RBRACE = "}"
  LPAREN = "("
  RPAREN = ")"
  LBRACKET = "["
  RBRACKET = "]"
  COMMA = ","
  COLON = ":"
  EQUAL = "="
