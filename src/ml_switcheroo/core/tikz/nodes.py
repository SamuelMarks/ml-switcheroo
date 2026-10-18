"""TikZ Concrete Syntax Tree (CST) Nodes.

This module defines the data structures for representing TikZ source code.
It follows a similar philosophy to LibCST, where nodes own their
string representation via a ``to_text()`` method, allowing for precise control
over formatting (whitespace, indentation) during code generation.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ml_switcheroo.core.markup.cst import MarkupNode


class TikzBaseNode(MarkupNode, ABC):
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
  """Represent non-semantic textual elements (whitespace, newlines, comments)."""

  content: str = ""
  """Execute raw text content."""

  kind: str = "whitespace"
  """Either 'whitespace' or 'comment'."""

  def __init__(
    self,
    content: str = "",
    kind: str = "whitespace",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TriviaNode.

    Args:
        content: Raw trivia text content.
        kind: Trivia kind ('whitespace' or 'comment').
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__()
    self.content = content
    self.kind = kind

  def emit(self, indent_level: int = 0) -> str:
    """Return the raw whitespace/comment content verbatim.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The raw trivia content.
    """
    return self.content


@dataclass
class TikzOption(TikzBaseNode):
  """Represent a TikZ option like ``[draw=black]`` or ``[circle]``."""

  key: str = ""
  """Option key."""

  value: Optional[str] = None
  """Optional value for key-value pairs."""

  def __init__(
    self,
    key: str = "",
    value: Optional[str] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TikzOption.

    Args:
        key: Option key.
        value: Optional value for key-value pairs.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__()
    self.key = key
    self.value = value

  def emit(self, indent_level: int = 0) -> str:
    """Return ``key=value`` or just ``key``.

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
  """Represent text with basic LaTeX styling inside TikZ."""

  content: str = ""
  """Execute raw text content."""

  bold: bool = False
  """Whether to render the text in bold."""

  italic: bool = False
  """Whether to render the text in italics."""

  def __init__(
    self,
    content: str = "",
    bold: bool = False,
    italic: bool = False,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TikzTextNode.

    Args:
        content: Raw text content.
        bold: Whether to render in bold.
        italic: Whether to render in italics.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__()
    self.content = content
    self.bold = bold
    self.italic = italic

  def emit(self, indent_level: int = 0) -> str:
    """Emit the text node with styling.

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
  r"""Represent an HTML-like table structure used inside TikZ Node labels.

  Uses LaTeX tabular environment syntax.

  Example::

      \\begin{tabular}{c}
          \\textbf{LayerName} \\\\
          param: val
      \\end{tabular}
  """

  rows: List[List[Any]] = field(default_factory=list)
  """List of rows, where each row is a list of cell items."""

  align: str = "c"
  """Column alignment (c=center, l=left, r=right)."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  """Trivia elements appearing before the table environment."""

  trailing_trivia: List[TriviaNode] = field(default_factory=list)
  """Trivia elements appearing after the table environment."""

  def __init__(
    self,
    rows: Optional[List[List[Any]]] = None,
    align: str = "c",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TikzTable.

    Args:
        rows: List of rows.
        align: Column alignment.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__()
    self.rows = rows if rows is not None else []
    self.align = align
    self.leading_trivia = leading_trivia if leading_trivia is not None else []
    self.trailing_trivia = trailing_trivia if trailing_trivia is not None else []

  def emit(self, indent_level: int = 0) -> str:
    """Render the tabular environment string.

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
  r"""Represent a ``\\node`` command.

  Structure::

      \\node [options] (id) at (x, y) {label_content};
  """

  node_id: str = ""
  """Unique identifier for the node (used for edges)."""

  x: float = 0.0
  """X Coordinate."""

  y: float = 0.0
  """Y Coordinate."""

  content: Any = ""
  """Inner content (Text or Table)."""

  options: List[TikzOption] = field(default_factory=list)
  """List of TikZ options."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace/Comments before the node command."""

  trailing_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace/Comments after the node command."""

  def __init__(
    self,
    node_id: str = "",
    x: float = 0.0,
    y: float = 0.0,
    content: Any = "",
    options: Optional[List[TikzOption]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TikzNode.

    Args:
        node_id: Unique node identifier.
        x: X coordinate.
        y: Y coordinate.
        content: Inner text or TikzTable.
        options: List of TikZ options.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__()
    self.node_id = node_id
    self.x = x
    self.y = y
    self.content = content
    self.options = options if options is not None else []
    self.leading_trivia = leading_trivia if leading_trivia is not None else []
    self.trailing_trivia = trailing_trivia if trailing_trivia is not None else []

  def emit(self, indent_level: int = 0) -> str:
    """Construct the full node command string.

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
  r"""Represent a ``\\draw`` command connecting two nodes.

  Structure::

      \\draw [options] (src) -- (tgt);
  """

  source_id: str = ""
  """Source node ID."""

  target_id: str = ""
  """Target node ID."""

  options: List[TikzOption] = field(default_factory=list)
  """List of styling options."""

  connector: str = "--"
  """Connector style (e.g. ``--`` or ``->``)."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace before the draw command."""

  trailing_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace/Comments after the draw command."""

  def __init__(
    self,
    source_id: str = "",
    target_id: str = "",
    options: Optional[List[TikzOption]] = None,
    connector: str = "--",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TikzEdge.

    Args:
        source_id: Source node ID.
        target_id: Target node ID.
        options: List of styling options.
        connector: Connector style.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__()
    self.source_id = source_id
    self.target_id = target_id
    self.options = options if options is not None else []
    self.connector = connector
    self.leading_trivia = leading_trivia if leading_trivia is not None else []
    self.trailing_trivia = trailing_trivia if trailing_trivia is not None else []

  def emit(self, indent_level: int = 0) -> str:
    """Construct the draw command string.

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
  r"""Execute root container representing the ``tikzpicture`` environment.

  Structure::

      \\begin{tikzpicture}
          ... children ...
      \\end{tikzpicture}
  """

  children: List[Any] = field(default_factory=list)
  """List of nodes, edges, comments, and trivia."""

  options: List[TikzOption] = field(default_factory=list)
  """Global environment options."""

  leading_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace/Comments before the tikzpicture environment."""

  trailing_trivia: List[TriviaNode] = field(default_factory=list)
  """Whitespace/Comments after the tikzpicture environment."""

  def __init__(
    self,
    children: Optional[List[Any]] = None,
    options: Optional[List[TikzOption]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TikzGraph.

    Args:
        children: List of child nodes, edges, and trivia.
        options: Global environment options.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__()
    self.children = children if children is not None else []
    self.options = options if options is not None else []
    self.leading_trivia = leading_trivia if leading_trivia is not None else []
    self.trailing_trivia = trailing_trivia if trailing_trivia is not None else []

  def emit(self, indent_level: int = 0) -> str:
    """Construct the complete environment string.

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
