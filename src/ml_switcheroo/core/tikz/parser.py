"""
TikZ Parser (Lexer & Logical Reconstruction).

This module provides the `TikzParser` which consumes raw LaTeX/TikZ source code
and reconstructs the `LogicalGraph` representation. It effectively reverses
the operation of the `TikzEmitter`.

Capabilities:
1.  **Tokenization**: Regex-based lexer for LaTeX commands, groups, and options.
2.  **Structural Parsing**: Identifies nodes, edges, and environments via recursive descent.
3.  **Metadata Extraction**: Parses HTML-like tabular environments embedded in
    node labels to recover layer hyperparameters (e.g., kernel size, stride).
4.  **Graph Reconstruction**: Builds a `LogicalGraph` object compatible with
    the rest of the transpiler pipeline.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, List, Optional, Tuple, Dict

from ml_switcheroo.core.tikz.analyser import LogicalGraph, LogicalNode, LogicalEdge


class TokenKind(Enum):
  """Enumeration of LaTeX/TikZ token types."""

  COMMAND = auto()  # \node, \draw, \textbf
  LBRACE = auto()  # {
  RBRACE = auto()  # }
  LBRACKET = auto()  # [
  RBRACKET = auto()  # ]
  LPAREN = auto()  # (
  RPAREN = auto()  # )
  SEMICOLON = auto()  # ;
  ARROW = auto()  # -- or ->
  WORD = auto()  # Identifiers, values
  NUMBER = auto()  # 1.5, -2
  COMMENT = auto()  # % ...
  WHITESPACE = auto()  # space, tab, newline
  EOF = auto()


@dataclass
class Token:
  """A lexical unit with position info."""

  kind: TokenKind
  text: str
  line: int
  col: int


class TikzLexer:
  """
  Regex-based tokenizer for TikZ source code.

  Splits raw LaTeX strings into a stream of typed Tokens, handling
  symbols, commands, strings, and whitespace.
  """

  # Order matters: Specific patterns before general ones
  PATTERNS = [
    (TokenKind.COMMENT, r"%.*"),
    (TokenKind.COMMAND, r"\\[a-zA-Z]+"),
    (TokenKind.ARROW, r"--|->"),
    (TokenKind.LBRACE, r"\{"),
    (TokenKind.RBRACE, r"\}"),
    (TokenKind.LBRACKET, r"\["),
    (TokenKind.RBRACKET, r"\]"),
    (TokenKind.LPAREN, r"\("),
    (TokenKind.RPAREN, r"\)"),
    (TokenKind.SEMICOLON, r";"),
    (TokenKind.NUMBER, r"-?\d+(?:\.\d+)?"),
    # Catch-all for words/identifiers/symbols not caught above
    # Includes colons, underscores for now as part of 'words'
    (TokenKind.WORD, r"[a-zA-Z0-9_:,\.\\\>\<=]+"),
    (TokenKind.WHITESPACE, r"\s+"),
  ]

  def __init__(self, text: str):
    """Initialize the lexer with input text."""
    self.text = text
    self.pos = 0
    self.line = 1
    self.col = 1
    self._tokens: List[Token] = []

  def tokenize(self) -> List[Token]:
    """
    Converts the full string into a list of Tokens.

    Returns:
        List[Token]: Sequence of tokens, excluding comments and whitespace.
                     Ends with an EOF token.
    """
    while self.pos < len(self.text):
      match = None
      for kind, pattern in self.PATTERNS:
        regex = re.compile(pattern)
        match = regex.match(self.text, self.pos)
        if match:
          text = match.group(0)
          # Create token (skip whitespace/comments)
          if kind not in (TokenKind.WHITESPACE, TokenKind.COMMENT):
            self._tokens.append(Token(kind, text, self.line, self.col))

          # Update tracking
          newlines = text.count("\n")
          self.line += newlines
          if newlines > 0:
            self.col = len(text) - text.rfind("\n")
          else:
            self.col += len(text)

          self.pos = match.end()
          break

      if not match:
        # Skip unknown char (robustness)
        # In a real parser we might raise SyntaxError, but here we skip to be robust
        self.pos += 1
        self.col += 1

    self._tokens.append(Token(TokenKind.EOF, "", self.line, self.col))
    return self._tokens


class TikzParser:
  """
  Parses tokenized TikZ code into a LogicalGraph.

  This parser implements a recursive descent strategy tailored to the specific
  TikZ subset produced by the `TikzEmitter`. It is not a general-purpose
  TeX parser.
  """

  def __init__(self, text: str):
    """Initialize parser and tokenize input."""
    self.lexer = TikzLexer(text)
    self.tokens = self.lexer.tokenize()
    self.pos = 0
    self.graph = LogicalGraph()

  def parse(self) -> LogicalGraph:
    """
    Main entry point. Iterates top-level commands.

    Returns:
        LogicalGraph: The reconstructed graph extracted from the visual definition.
    """
    while not self._is_eof():
      token = self._peek()

      # Skip top-level environments like \begin{tikzpicture}
      if token.kind == TokenKind.COMMAND:
        if token.text == r"\begin":
          self._consume()  # \begin
          self._parse_braced_group()  # {tikzpicture}
          self._optional_bracket_group()  # [options]
          continue
        elif token.text == r"\end":
          self._consume()
          self._parse_braced_group()
          continue
        elif token.text == r"\node":
          self._parse_node()
          continue
        elif token.text == r"\draw":
          self._parse_edge()
          continue

      self._consume()

    return self.graph

  # --- Parser Primitives ---

  def _peek(self, offset: int = 0) -> Token:
    """Look ahead at a token without consumption."""
    idx = self.pos + offset
    if idx >= len(self.tokens):
      return self.tokens[-1]
    return self.tokens[idx]

  def _consume(self) -> Token:
    """Consume and return the current token."""
    token = self._peek()
    self.pos += 1
    return token

  def _match(self, kind: TokenKind) -> bool:
    """Check if the current token matches a specific kind."""
    return self._peek().kind == kind

  def _expect(self, kind: TokenKind) -> Token:
    """
    Consume the current token if it matches kind, else raise error.
    """
    if not self._match(kind):
      cur = self._peek()
      raise SyntaxError(f"Expected {kind}, got {cur.kind} ('{cur.text}') at line {cur.line}")
    return self._consume()

  def _is_eof(self) -> bool:
    """Check if End of File reached."""
    return self._peek().kind == TokenKind.EOF

  def _parse_braced_group(self) -> List[Token]:
    """
    Consumes tokens inside `{ ... }`, handling nesting.

    Returns:
        List[Token]: The tokens inside the braces.
    """
    self._expect(TokenKind.LBRACE)
    content = []
    depth = 1
    while depth > 0 and not self._is_eof():
      tk = self._consume()
      if tk.kind == TokenKind.LBRACE:
        depth += 1
      elif tk.kind == TokenKind.RBRACE:
        depth -= 1

      if depth > 0:
        content.append(tk)
    return content

  def _optional_bracket_group(self) -> List[Token]:
    """
    Consumes `[...]` group if present.

    Returns:
        List[Token]: The tokens inside brackets, or empty list if brackets specific not found.
    """
    if self._match(TokenKind.LBRACKET):
      self._consume()
      content = []
      while not self._match(TokenKind.RBRACKET) and not self._is_eof():
        content.append(self._consume())
      self._consume()  # ]
      return content
    return []

  # --- Feature Parsing ---

  def _parse_node(self) -> None:
    """
    Parses a `\\node` command and adds a LogicalNode to the graph.
    Expects format: `\\node [options] (id) at (x,y) {content};`.
    """
    self._expect(TokenKind.COMMAND)  # \node

    # Options [fill=...]
    self._optional_bracket_group()

    # ID (id)
    if self._match(TokenKind.LPAREN):
      self._consume()
      node_id_tk = self._expect(TokenKind.WORD)
      self._expect(TokenKind.RPAREN)
      node_id = node_id_tk.text
    else:
      # Nodes without IDs are usually aux/labels. We skip them to avoid noise.
      self._scan_until_semicolon()
      return

    # Position "at (x, y)"
    # We parse but discard position for LogicalGraph purposes
    if self._peek().text == "at":
      self._consume()
      self._consume()  # (
      # Consume coordinates (can be numbers, words, commas)
      while not self._match(TokenKind.RPAREN) and not self._is_eof():
        self._consume()
      self._consume()  # )

    # Content { ... }
    content_tokens = self._parse_braced_group()

    # Parse content for metadata using tabular extraction heuristics
    kind, metadata = self._extract_metadata(content_tokens)

    # Build Logical Node
    node = LogicalNode(id=node_id, kind=kind, metadata=metadata)
    self.graph.nodes.append(node)

    # Trailing semicolon
    if self._match(TokenKind.SEMICOLON):
      self._consume()

  def _parse_edge(self) -> None:
    """
    Parses a `\\draw` command and adds a LogicalEdge to the graph.
    Expects format: `\\draw [opts] (src) -- (tgt);`.
    """
    self._expect(TokenKind.COMMAND)  # \draw
    self._optional_bracket_group()

    # Source
    self._expect(TokenKind.LPAREN)
    src = self._expect(TokenKind.WORD).text
    self._expect(TokenKind.RPAREN)

    # Connector
    if self._match(TokenKind.ARROW):
      self._consume()
    else:
      # Unexpected edge format, skip
      self._scan_until_semicolon()
      return

    # Target
    self._expect(TokenKind.LPAREN)
    tgt = self._expect(TokenKind.WORD).text
    self._expect(TokenKind.RPAREN)

    # Add Edge
    self.graph.edges.append(LogicalEdge(source=src, target=tgt))

    # Trailing semicolon
    if self._match(TokenKind.SEMICOLON):
      self._consume()

  def _scan_until_semicolon(self) -> None:
    """Helper to consume tokens until a semicolon is found (Error Recovery)."""
    while not self._match(TokenKind.SEMICOLON) and not self._is_eof():
      self._consume()
    if self._match(TokenKind.SEMICOLON):
      self._consume()

  def _extract_metadata(self, tokens: List[Token]) -> Tuple[str, Dict[str, str]]:
    """
    Parses the node label content to extract Op Kind and Config.

    Expected structure is a LaTeX tabular:
        Kind (Row 1)
        ID   (Row 2, ignored)
        key: value (Row 3+)

    Args:
        tokens: The list of raw tokens inside the node body.

    Returns:
        Tuple of (Kind String, Metadata Dict).
    """
    # Linear scan for text tokens.
    # We ignore LaTeX formatting macros like \textbf, \textit, \begin, \end, &, \\
    # We rely on positional logic: 1st meaningful word is Kind. Key-Values follow.

    words = []
    for tk in tokens:
      if tk.kind == TokenKind.WORD or tk.kind == TokenKind.NUMBER:
        clean = tk.text.replace(r"\_", "_")  # Unescape underscore
        words.append(clean)
      elif tk.kind == TokenKind.COMMAND:
        pass  # Ignore formatting commands

    # Filter out 'tabular' and 'c' which are environment artifacts
    content = [w for w in words if w not in ["tabular", "c", "l", "r"]]

    if not content:
      return ("Unknown", {})

    kind = content[0]
    metadata = {}

    # The rest should be key-value pairs?
    # The emitter puts ID in italics as row 2.
    # Row 1: Kind. Row 2: ID. Row 3+: Metadata.

    start_idx = 2  # Skip Kind and ID
    if len(content) <= start_idx:
      # Just return what we have
      return (kind, metadata)

    # Fallback to key/value extraction from token text list
    # Look for tokens ending in ":" to identify keys
    current_key = None

    for w in content[start_idx:]:
      if w.endswith(":"):
        current_key = w[:-1]
      elif current_key:
        metadata[current_key] = w
        current_key = None

    return kind, metadata
