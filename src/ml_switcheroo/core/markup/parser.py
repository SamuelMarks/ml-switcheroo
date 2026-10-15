"""Shared Parser Infrastructure for Markup DSLs.

Provides tokenization utilities and recursive descent parser base classes
shared across HTML, LaTeX, and TikZ frontends.
"""

import re
from typing import List, Optional, Tuple


class Token:
  """A lexical token produced during markup tokenization."""

  def __init__(self, type_: str, value: str, pos: int = 0) -> None:
    """Initialize a lexical token.

    Args:
        type_: Symbolic token category name (e.g., 'TAG', 'IDENT').
        value: Raw matched substring from the source text.
        pos: Zero-based character offset in the source string.
    """
    self.type = type_
    self.value = value
    self.pos = pos

  def __repr__(self) -> str:
    """Return a developer-friendly string representation of the token.

    Returns:
        str: Formatted Token representation string.
    """
    return f"Token({self.type}, {repr(self.value)})"


class BaseTokenizer:
  """Base generic regex-based tokenizer for markup DSLs."""

  def __init__(self, token_specification: List[Tuple[str, str]]) -> None:
    """Initialize the tokenizer with named regular expression rules.

    Args:
        token_specification: Ordered list of (token_name, regex_pattern) tuples.
    """
    self.token_specification = token_specification
    self.tok_regex = re.compile("|".join("(?P<%s>%s)" % pair for pair in token_specification), re.MULTILINE | re.DOTALL)

  def tokenize(self, text: str) -> List[Token]:
    """Tokenize an input string into a list of non-skipped Token objects.

    Args:
        text: Input source string to lex.

    Returns:
        List[Token]: Sequence of extracted tokens.
    """
    tokens = []
    for mo in self.tok_regex.finditer(text):
      kind = mo.lastgroup
      value = mo.group(kind) if kind else ""
      if kind == "SKIP" or not kind:
        continue
      tokens.append(Token(kind, value, mo.start()))
    return tokens


class RecursiveDescentParser:
  """Base recursive descent parser infrastructure over a stream of Tokens."""

  def __init__(self, tokens: List[Token]) -> None:
    """Initialize the recursive descent parser.

    Args:
        tokens: Ordered list of lexical tokens to parse.
    """
    self.tokens = tokens
    self.pos = 0

  @property
  def current(self) -> Optional[Token]:
    """Return the token at the current cursor position, or None if at EOF.

    Returns:
        Optional[Token]: Current token or None.
    """
    if self.pos < len(self.tokens):
      return self.tokens[self.pos]
    return None

  def advance(self) -> Optional[Token]:
    """Consume and return the current token, advancing the cursor by one.

    Returns:
        Optional[Token]: The consumed token, or None if at EOF.
    """
    tok = self.current
    self.pos += 1
    return tok

  def match(self, token_type: str) -> bool:
    """Consume the current token if its type matches ``token_type``.

    Args:
        token_type: Expected token type identifier.

    Returns:
        bool: True if the current token matched and was consumed, False otherwise.
    """
    if self.current and self.current.type == token_type:
      self.advance()
      return True
    return False

  def expect(self, token_type: str) -> Token:
    """Consume and return the current token if it matches ``token_type``, else raise ValueError.

    Args:
        token_type: Required token type identifier.

    Returns:
        Token: The matched token.

    Raises:
        ValueError: If the current token is None or has a different type.
    """
    tok = self.current
    if not tok or tok.type != token_type:
      raise ValueError(f"Expected {token_type} but got {tok.type if tok else 'EOF'}")
    self.advance()
    return tok
