"""TikZ Concrete Syntax Tree (CST) Nodes.

This module defines the data structures for representing TikZ source code.
It follows a similar philosophy to LibCST, where nodes own their
string representation via a ``to_text()`` method, allowing for precise control
over formatting (whitespace, indentation) during code generation.
"""

import typing
from abc import ABC

import pydantic

from ml_switcheroo.core.cst.base import CstFieldValue, TriviaInput
from ml_switcheroo.core.markup.cst import MarkupNode
from ml_switcheroo.core.markup.cst import TriviaNode as MarkupTriviaNode


class TikzBaseNode(MarkupNode, ABC):
  """Abstract base class for all TikZ CST nodes."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize TikzBaseNode.

    Args:
        leading_trivia: Optional leading trivia preceding this node.
        trailing_trivia: Optional trailing trivia following this node.
        **kwargs: Subclass field values forwarded to MarkupNode.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)

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


class TriviaNode(MarkupTriviaNode, TikzBaseNode):
  """Represent non-semantic textual elements (whitespace, newlines, comments)."""

  def __init__(
    self,
    content: str = "",
    kind: str = "whitespace",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TriviaNode.

    Args:
        content: Raw trivia text content.
        kind: Trivia kind ('whitespace' or 'comment').
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      content=content,
      kind=kind,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )


class TikzOption(TikzBaseNode):
  """Represent a TikZ option like ``[draw=black]`` or ``[circle]``."""

  key: str = ""
  """Option key."""

  value: str | None = None
  """Optional value for key-value pairs."""

  def __init__(
    self,
    key: str = "",
    value: str | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TikzOption.

    Args:
        key: Option key.
        value: Optional value for key-value pairs.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(key=key, value=value, leading_trivia=leading_trivia, trailing_trivia=trailing_trivia)

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
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TikzTextNode.

    Args:
        content: Raw text content.
        bold: Whether to render in bold.
        italic: Whether to render in italics.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      content=content,
      bold=bold,
      italic=italic,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )

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


TikzCellItem: typing.TypeAlias = TikzBaseNode | str | int | float


class TikzTable(TikzBaseNode):
  r"""Represent an HTML-like table structure used inside TikZ Node labels.

  Uses LaTeX tabular environment syntax.

  Example::

      \\begin{tabular}{c}
          \\textbf{LayerName} \\\\
          param: val
      \\end{tabular}
  """

  rows: list[list[TikzCellItem]] = pydantic.Field(default_factory=list)
  """List of rows, where each row is a list of cell items."""

  align: str = "c"
  """Column alignment (c=center, l=left, r=right)."""

  def __init__(
    self,
    rows: typing.Sequence[typing.Sequence[TikzCellItem]] | None = None,
    align: str = "c",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TikzTable.

    Args:
        rows: List of rows.
        align: Column alignment.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      rows=[list(r) for r in rows] if rows is not None else [],
      align=align,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )

  def emit(self, indent_level: int = 0) -> str:
    """Render the tabular environment string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The formatted LaTeX tabular string.
    """
    # We ignore indent_level for pure roundtripping if trivia exists, but it's kept for API compatibility.
    parts = [self.leading_trivia_text()]

    parts.append(f"\\begin{{tabular}}{{{self.align}}}")
    # For now, to keep it simple, we just format the rows standardly
    # In a full CST this would also have row trivia.
    for row in self.rows:
      emitted_row = [cell.emit() if isinstance(cell, TikzBaseNode) else str(cell) for cell in row]
      parts.append(" & ".join(emitted_row) + r" \\")
    parts.append(r"\end{tabular}")

    parts.append(self.trailing_trivia_text())
    return "".join(parts)


class TikzNode(TikzBaseNode):
  r"""Represent a ``\\node`` command.

  Structure::

      \\node [options] (id) at (x, y) {label_content};
  """

  node_id: str = ""
  """Unique identifier for the node (used for edges)."""

  x: int | float = 0
  """X Coordinate."""

  y: int | float = 0
  """Y Coordinate."""

  content: TikzBaseNode | str = ""
  """Inner content (Text or Table)."""

  options: list[TikzOption] = pydantic.Field(default_factory=list)
  """List of TikZ options."""

  def __init__(
    self,
    node_id: str = "",
    x: int | float = 0,
    y: int | float = 0,
    content: TikzBaseNode | str = "",
    options: typing.Sequence[TikzOption] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
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
    super().__init__(
      node_id=node_id,
      x=x,
      y=y,
      content=content,
      options=list(options) if options is not None else [],
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )

  def emit(self, indent_level: int = 0) -> str:
    """Construct the full node command string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The full LaTeX/TikZ node command.
    """
    parts = [self.leading_trivia_text(), r"\node"]

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f" [{opts_str}]")

    parts.append(f" ({self.node_id})")
    parts.append(f" at ({self.x}, {self.y})")

    content_str = self.content.emit(indent_level) if isinstance(self.content, TikzBaseNode) else str(self.content)

    parts.append(" {")
    parts.append(content_str)
    parts.append("};")

    parts.append(self.trailing_trivia_text())
    return "".join(parts)


class TikzEdge(TikzBaseNode):
  r"""Represent a ``\\draw`` command connecting two nodes.

  Structure::

      \\draw [options] (src) -- (tgt);
  """

  source_id: str = ""
  """Source node ID."""

  target_id: str = ""
  """Target node ID."""

  options: list[TikzOption] = pydantic.Field(default_factory=list)
  """List of styling options."""

  connector: str = "--"
  """Connector style (e.g. ``--`` or ``->``)."""

  def __init__(
    self,
    source_id: str = "",
    target_id: str = "",
    options: typing.Sequence[TikzOption] | None = None,
    connector: str = "--",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
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
    super().__init__(
      source_id=source_id,
      target_id=target_id,
      options=list(options) if options is not None else [],
      connector=connector,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )

  def emit(self, indent_level: int = 0) -> str:
    """Construct the draw command string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The full LaTeX/TikZ draw command.
    """
    parts = [self.leading_trivia_text(), r"\draw"]

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f" [{opts_str}]")

    parts.append(f" ({self.source_id})")
    parts.append(f" {self.connector}")
    parts.append(f" ({self.target_id});")

    parts.append(self.trailing_trivia_text())
    return "".join(parts)


class TikzGraph(TikzBaseNode):
  r"""Execute root container representing the ``tikzpicture`` environment.

  Structure::

      \\begin{tikzpicture}
          ... children ...
      \\end{tikzpicture}
  """

  children: list[TikzBaseNode] = pydantic.Field(default_factory=list)
  """List of nodes, edges, comments, and trivia."""

  options: list[TikzOption] = pydantic.Field(default_factory=list)
  """Global environment options."""

  def __init__(
    self,
    children: typing.Sequence[TikzBaseNode] | None = None,
    options: typing.Sequence[TikzOption] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TikzGraph.

    Args:
        children: List of child nodes, edges, and trivia.
        options: Global environment options.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      children=list(children) if children is not None else [],
      options=list(options) if options is not None else [],
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )

  def emit(self, indent_level: int = 0) -> str:
    """Construct the complete environment string.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The complete formatted tikzpicture environment.
    """
    parts = [self.leading_trivia_text()]

    if self.options:
      opts_str = ", ".join([o.emit() for o in self.options])
      parts.append(f"\\begin{{tikzpicture}}[{opts_str}]")
    else:
      parts.append(r"\begin{tikzpicture}")

    parts.extend(child.emit(indent_level) for child in self.children)

    parts.append(r"\end{tikzpicture}")
    parts.append(self.trailing_trivia_text())
    return "".join(parts)
