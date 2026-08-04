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
    """Render this node to its string representation.

    Returns:
        str: The full structured text of this node.
    """
    return self.emit(0)


@dataclass
class TriviaNode(TikzBaseNode):
  """Represents non-semantic textual elements (whitespace, newlines, comments)."""

  content: str
  """The raw text content."""

  kind: str = "whitespace"
  """Either 'whitespace' or 'comment'."""

  def emit(self, indent_level: int = 0) -> str:
    """Returns the raw whitespace/comment content verbatim.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The raw trivia content.
    """
    return self.content


@dataclass
class TikzOption(TikzBaseNode):
  """Represents a TikZ option like ``[draw=black]`` or ``[circle]``."""

  key: str
  """Option key."""

  value: Optional[str] = None
  """Optional value for key-value pairs."""

  def emit(self, indent_level: int = 0) -> str:
    """Returns ``key=value`` or just ``key``.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The formatted TikZ option.
    """
    if self.value:
      return f"{self.key}={self.value}"
    return self.key


@dataclass
class TikzTextNode(TikzBaseNode):
  """Represents text with basic LaTeX styling inside TikZ."""

  content: str
  """The raw text content."""

  bold: bool = False
  """Whether to render the text in bold."""

  italic: bool = False
  """Whether to render the text in italics."""

  def emit(self, indent_level: int = 0) -> str:
    """Emits the text node with styling.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The styled LaTeX text.
    """
    res = self.content
    if self.italic:
      res = f"\\textit{{{res}}}"
    if self.bold:
      res = f"\\textbf{{{res}}}"
    return res


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

  rows: List[List[Union[str, TikzTextNode]]] = field(default_factory=list)
  """List of rows, where each row is a list of cell items."""

  align: str = "c"
  """Column alignment (c=center, l=left, r=right)."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  """Trivia elements appearing before the table environment."""

  trailing_trivia: List[TriviaNode] = field(default_factory=list)
  """Trivia elements appearing after the table environment."""

  def emit(self, indent_level: int = 0) -> str:
    """Renders the tabular environment string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The formatted LaTeX tabular string.
    """
    # We ignore indent_level for pure roundtripping if trivia exists, but it's kept for API compatibility.
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())

    parts.append(f"\\begin{{tabular}}{{{self.align}}}")
    # For now, to keep it simple, we just format the rows standardly
    # In a full CST this would also have row trivia.
    for row in self.rows:
      emitted_row = []
      for cell in row:
        emitted_row.append(cell.emit() if isinstance(cell, TikzBaseNode) else str(cell))
      parts.append(" & ".join(emitted_row) + r" \\")
    parts.append(r"\end{tabular}")

    for t in self.trailing_trivia:
      parts.append(t.emit())
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
  """Whitespace/Comments after the node command."""

  def emit(self, indent_level: int = 0) -> str:
    """Constructs the full node command string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The full LaTeX/TikZ node command.
    """
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())

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
      parts.append(t.emit())
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
  """Whitespace/Comments after the draw command."""

  def emit(self, indent_level: int = 0) -> str:
    """Constructs the draw command string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The full LaTeX/TikZ draw command.
    """
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())

    parts.append(r"\draw")

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f" [{opts_str}]")

    parts.append(f" ({self.source_id})")
    parts.append(f" {self.connector}")
    parts.append(f" ({self.target_id});")

    for t in self.trailing_trivia:
      parts.append(t.emit())
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
  """Whitespace/Comments before the tikzpicture environment."""

  trailing_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace/Comments after the tikzpicture environment."""

  def emit(self, indent_level: int = 0) -> str:
    """Constructs the complete environment string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The complete formatted tikzpicture environment.
    """
    parts = []
    for t in self.leading_trivia:
      parts.append(t.emit())

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f"\\begin{{tikzpicture}}[{opts_str}]")
    else:
      parts.append(r"\begin{tikzpicture}")

    for child in self.children:
      parts.append(child.emit(indent_level))

    parts.append(r"\end{tikzpicture}")

    for t in self.trailing_trivia:
      parts.append(t.emit())
    return "".join(parts)
