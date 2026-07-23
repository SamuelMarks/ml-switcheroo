"""TikZ Concrete Syntax Tree (CST) Nodes.

This module defines the data structures for representing TikZ source code.
It follows a similar philosophy to LibCST, where nodes own their
string representation via a ``to_text()`` method, allowing for precise control
over formatting (whitespace, indentation) during code generation.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TikzBaseNode(ABC):
  """Abstract base class for all TikZ CST nodes."""

  def emit(self, indent_level: int = 0) -> str:
    """Render this node to its string representation with indentation.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The structured TikZ/LaTeX source code.
    """
    return ""

  def to_text(self) -> str:
    """Render this node to its string representation."""
    return self.emit(0)


@dataclass
class TriviaNode(TikzBaseNode):
  """Represents non-semantic textual elements (whitespace, newlines, comments)."""

  content: str
  """The raw text content."""

  kind: str = "whitespace"
  """Either 'whitespace' or 'comment'."""

  def emit(self, indent_level: int = 0) -> str:
    """Returns the raw whitespace/comment content verbatim."""
    return self.content


@dataclass
class TikzOption(TikzBaseNode):
  """Represents a TikZ option like ``[draw=black]`` or ``[circle]``."""

  key: str
  """Option key."""

  value: Optional[str] = None
  """Optional value for key-value pairs."""

  def emit(self, indent_level: int = 0) -> str:
    """Returns ``key=value`` or just ``key``."""
    if self.value:
      return f"{self.key}={self.value}"
    return self.key


@dataclass
class TikzTable(TikzBaseNode):
  r"""Represents an HTML-like table structure used inside TikZ Node labels.

  Uses LaTeX tabular environment syntax.

  Example::

      \\begin{tabular}{c}
          \\textbf{LayerName} \\\\
          param: val
      \\end{tabular}
  """

  rows: List[List[str]] = field(default_factory=list)
  """List of rows, where each row is a list of cell strings."""

  align: str = "c"
  """Column alignment (c=center, l=left, r=right)."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  trailing_trivia: List[TriviaNode] = field(default_factory=list)

  def emit(self, indent_level: int = 0) -> str:
    """Renders the tabular environment string."""
    # We ignore indent_level for pure roundtripping if trivia exists, but it's kept for API compatibility.
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())  # pragma: no cover

    parts.append(f"\\begin{{tabular}}{{{self.align}}}")
    # For now, to keep it simple, we just format the rows standardly
    # In a full CST this would also have row trivia.
    for row in self.rows:
      parts.append(" & ".join(row) + r" \\")
    parts.append(r"\end{tabular}")

    for t in self.trailing_trivia:
      parts.append(t.emit())  # pragma: no cover
    return "".join(parts)


@dataclass
class TikzNode(TikzBaseNode):
  r"""Represents a ``\\node`` command.

  Structure::

      \\node [options] (id) at (x, y) {label_content};
  """

  node_id: str
  """Unique identifier for the node (used for edges)."""

  x: float
  """X Coordinate."""

  y: float
  """Y Coordinate."""

  content: Union[str, TikzTable]
  """Inner content (Text or Table)."""

  options: List[TikzOption] = field(default_factory=list)
  """List of TikZ options."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace/Comments before the node command."""
  trailing_trivia: List[TriviaNode] = field(default_factory=list)

  def emit(self, indent_level: int = 0) -> str:
    """Constructs the full node command string."""
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())  # pragma: no cover

    parts.append(r"\node")

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f" [{opts_str}]")

    parts.append(f" ({self.node_id})")
    parts.append(f" at ({self.x}, {self.y})")

    content_str = self.content.emit(indent_level) if isinstance(self.content, TikzBaseNode) else str(self.content)

    parts.append(" {")
    parts.append(content_str)
    parts.append("};")

    for t in self.trailing_trivia:
      parts.append(t.emit())  # pragma: no cover
    return "".join(parts)


@dataclass
class TikzEdge(TikzBaseNode):
  r"""Represents a ``\\draw`` command connecting two nodes.

  Structure::

      \\draw [options] (src) -- (tgt);
  """

  source_id: str
  """Source node ID."""

  target_id: str
  """Target node ID."""

  options: List[TikzOption] = field(default_factory=list)
  """List of styling options."""

  connector: str = "--"
  """Connector style (e.g. ``--`` or ``->``)."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace before the draw command."""
  trailing_trivia: List[TriviaNode] = field(default_factory=list)

  def emit(self, indent_level: int = 0) -> str:
    """Constructs the draw command string."""
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())  # pragma: no cover

    parts.append(r"\draw")

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f" [{opts_str}]")

    parts.append(f" ({self.source_id})")
    parts.append(f" {self.connector}")
    parts.append(f" ({self.target_id});")

    for t in self.trailing_trivia:
      parts.append(t.emit())  # pragma: no cover
    return "".join(parts)


@dataclass
class TikzGraph(TikzBaseNode):
  r"""The root container representing the ``tikzpicture`` environment.

  Structure::

      \\begin{tikzpicture}
          ... children ...
      \\end{tikzpicture}
  """

  children: List[TikzBaseNode] = field(default_factory=list)
  """List of nodes, edges, comments, and trivia."""

  options: List[TikzOption] = field(default_factory=list)
  """Global environment options."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  trailing_trivia: List[TriviaNode] = field(default_factory=list)

  def emit(self, indent_level: int = 0) -> str:
    """Constructs the complete environment string."""
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())  # pragma: no cover

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f"\\begin{{tikzpicture}}[{opts_str}]")
    else:
      parts.append(r"\begin{tikzpicture}")

    for child in self.children:
      parts.append(child.emit(indent_level))

    parts.append(r"\end{tikzpicture}")

    for t in self.trailing_trivia:
      parts.append(t.emit())  # pragma: no cover
    return "".join(parts)
