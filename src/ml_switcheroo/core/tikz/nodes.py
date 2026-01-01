"""
TikZ Concrete Syntax Tree (CST) Nodes.

This module defines the data structures for representing TikZ source code.
It follows a similar philosophy to LibCST, where nodes own their
string representation via a ``to_text()`` method, allowing for precise control
over formatting (whitespace, indentation) during code generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TikzBaseNode(ABC):
  """Abstract base class for all TikZ CST nodes."""

  @abstractmethod
  def to_text(self) -> str:
    """
    Render this node to its string representation.

    Returns:
        str: The TikZ/LaTeX source code for this construct.
    """
    pass


@dataclass
class TriviaNode(TikzBaseNode):
  """
  Represents non-semantic textual elements (whitespace, newlines).
  """

  content: str
  """The raw whitespace content."""

  def to_text(self) -> str:
    """Returns the raw whitespace content."""
    return self.content


@dataclass
class TikzComment(TikzBaseNode):
  """
  Represents a LaTeX comment (e.g. ``% My Comment``).
  Includes the percent sign in the content or adds it during export.
  """

  text: str
  """The comment text."""

  trailing_newline: bool = True
  """Whether to append a newline after the comment."""

  def to_text(self) -> str:
    """Formats the comment with a leading percent sign."""
    clean = self.text.lstrip("%").strip()
    suffix = "\n" if self.trailing_newline else ""
    return f"% {clean}{suffix}"


@dataclass
class TikzOption(TikzBaseNode):
  """
  Represents a TikZ option like ``[draw=black]`` or ``[circle]``.
  """

  key: str
  """Option key."""

  value: Optional[str] = None
  """Optional value for key-value pairs."""

  def to_text(self) -> str:
    """Returns ``key=value`` or just ``key``."""
    if self.value:
      return f"{self.key}={self.value}"
    return self.key


@dataclass
class TikzTable(TikzBaseNode):
  """ 
    Represents an HTML-like table structure used inside TikZ Node labels. 
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

  def to_text(self) -> str:
    """Renders the tabular environment string."""
    lines = [f"\\begin{{tabular}}{{{self.align}}}"]
    for row in self.rows:
      # Join cells with &, append double-backslash for newline
      line_content = " & ".join(row) + r" \\"
      lines.append(f"    {line_content}")

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


@dataclass
class TikzNode(TikzBaseNode):
  """
  Represents a ``\\node`` command.

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

  def to_text(self) -> str:
    """Constructs the full node command string."""
    parts = []
    for t in self.leading_trivia:
      parts.append(t.to_text())

    parts.append(r"\node")

    if self.options:
      opts_str = ", ".join([o.to_text() for o in self.options])
      parts.append(f" [{opts_str}]")

    parts.append(f" ({self.node_id})")
    parts.append(f" at ({self.x}, {self.y})")

    content_str = self.content.to_text() if isinstance(self.content, TikzBaseNode) else self.content
    parts.append(" {")
    parts.append(content_str)
    parts.append("};")

    return "".join(parts)


@dataclass
class TikzEdge(TikzBaseNode):
  """
  Represents a ``\\draw`` command connecting two nodes.

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

  def to_text(self) -> str:
    """Constructs the draw command string."""
    parts = []
    for t in self.leading_trivia:
      parts.append(t.to_text())

    parts.append(r"\draw")

    if self.options:
      opts_str = ", ".join([o.to_text() for o in self.options])
      parts.append(f" [{opts_str}]")

    parts.append(f" ({self.source_id})")
    parts.append(f" {self.connector}")
    parts.append(f" ({self.target_id});")

    return "".join(parts)


@dataclass
class TikzGraph(TikzBaseNode):
  """
  The root container representing the ``tikzpicture`` environment.

  Structure::

      \\begin{tikzpicture}
          ... children ...
      \\end{tikzpicture}
  """

  children: List[TikzBaseNode] = field(default_factory=list)
  """List of nodes, edges, comments, and trivia."""

  options: List[TikzOption] = field(default_factory=list)
  """Global environment options."""

  def to_text(self) -> str:
    """Constructs the complete environment string."""
    lines = []

    # Open environment
    if self.options:
      opts_str = ", ".join([o.to_text() for o in self.options])
      lines.append(f"\\begin{{tikzpicture}}[{opts_str}]")
    else:
      lines.append(r"\begin{tikzpicture}")

    # Add children (indenting them)
    for child in self.children:
      child_text = child.to_text()
      # If child has multiple lines, indent all of them
      for line in child_text.splitlines():
        if line.strip():  # Indent non-empty lines
          lines.append(f"    {line}")
        else:
          lines.append("")  # Preserve empty lines

    # Close environment
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)
