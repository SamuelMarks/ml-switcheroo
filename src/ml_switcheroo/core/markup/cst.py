"""Common Markup Concrete Syntax Tree (CST) Nodes for Visual DSLs.

Defines the shared CST node hierarchy for HTML, LaTeX, and TikZ visualizations,
inheriting directly from :class:`ml_switcheroo.core.cst.base.CSTNode`.
"""

from ml_switcheroo.core.cst.base import CstFieldValue, CSTNode, TriviaInput


class MarkupNode(CSTNode):
  """Abstract base class for all markup DSL elements."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize MarkupNode.

    Args:
        leading_trivia: Optional leading trivia preceding this node.
        trailing_trivia: Optional trailing trivia following this node.
        **kwargs: Subclass field values forwarded to CSTNode.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)

  def emit(self, indent_level: int = 0) -> str:
    """Render this node to its string representation with indentation.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The structured markup source code.

    Raises:
        NotImplementedError: Must be implemented by concrete `MarkupNode` subclasses.
    """
    raise NotImplementedError

  def to_text(self) -> str:
    """Render this node to its string representation.

    Returns:
        str: The full structured text of this node.
    """
    return self.emit(0)


class TriviaNode(MarkupNode):
  """Represent non-semantic textual elements (whitespace, newlines, comments).

  Attributes:
      content: Raw trivia text content (`str`).
      kind: Trivia classification (`str`).
  """

  content: str = ""
  kind: str = "whitespace"

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
        kind: Trivia kind.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      content=content,
      kind=kind,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )

  def emit(self, indent_level: int = 0) -> str:
    """Emit raw trivia content without modification.

    Args:
        indent_level: Current indentation depth (unused for trivia).

    Returns:
        str: Raw trivia text content.
    """
    return self.content


class TextNode(MarkupNode):
  """Represent standard text content within markup.

  Attributes:
      content: Text content (`str`).
  """

  content: str = ""

  def __init__(
    self,
    content: str = "",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TextNode.

    Args:
        content: Text content.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      content=content,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
    )

  def emit(self, indent_level: int = 0) -> str:
    """Emit text content with leading and trailing trivia.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: Text string with leading and trailing trivia.
    """
    return f"{self.leading_trivia_text()}{self.content}{self.trailing_trivia_text()}"
