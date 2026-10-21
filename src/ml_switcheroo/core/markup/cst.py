"""Common Markup Concrete Syntax Tree (CST) Nodes for Visual DSLs.

Defines the shared AST node hierarchy for HTML, LaTeX, and TikZ visualizations.
"""

import typing

import pydantic

from ml_switcheroo.core.cst.base import CSTNode, Trivia


class MarkupNode(pydantic.BaseModel):
  """Abstract base class for all markup DSL elements."""

  model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  def __setattr__(self, name: str, value: typing.Hashable | typing.Sized) -> None:
    """Set attribute on MarkupNode, falling back to object.__setattr__ for dynamic attributes.

    Args:
        name: Attribute name.
        value: Attribute value.
    """
    try:
      super().__setattr__(name, value)
    except ValueError:
      object.__setattr__(self, name, value)

  def emit(self, indent_level: int = 0) -> str:
    """Render this node to its string representation with indentation.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The structured markup source code.
    """
    raise NotImplementedError

  def to_text(self) -> str:
    """Render this node to its string representation.

    Returns:
        str: The full structured text of this node.
    """
    return self.emit(0)


class TriviaNode(CSTNode, MarkupNode):
  """Represent non-semantic textual elements (whitespace, newlines, comments)."""

  content: str = ""
  kind: str = "whitespace"
  leading_trivia: list[Trivia] = pydantic.Field(default_factory=list)
  trailing_trivia: list[Trivia] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    content: str = "",
    kind: str = "whitespace",
    leading_trivia: list[Trivia] | None = None,
    trailing_trivia: list[Trivia] | None = None,
  ) -> None:
    """Initialize TriviaNode.

    Args:
        content: Raw trivia text content.
        kind: Trivia kind.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      content=content,
      kind=kind,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def emit(self, indent_level: int = 0) -> str:
    """Emit raw trivia content without modification.

    Args:
        indent_level: Current indentation depth (unused for trivia).

    Returns:
        str: Raw trivia text content.
    """
    return self.content


class TextNode(CSTNode, MarkupNode):
  """Represent standard text content within markup."""

  content: str = ""
  bold: bool = False
  italic: bool = False
  leading_trivia: list[Trivia] = pydantic.Field(default_factory=list)
  trailing_trivia: list[Trivia] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    content: str = "",
    bold: bool = False,
    italic: bool = False,
    leading_trivia: list[Trivia] | None = None,
    trailing_trivia: list[Trivia] | None = None,
  ) -> None:
    """Initialize TextNode.

    Args:
        content: Text content.
        bold: Bold flag.
        italic: Italic flag.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      content=content,
      bold=bold,
      italic=italic,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def emit(self, indent_level: int = 0) -> str:
    """Emit formatted text content with optional bold/italic styling and trivia.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: Formatted text string with leading and trailing trivia.
    """
    res = "".join(t.text for t in self.leading_trivia)
    val = self.content
    if self.bold:
      val = f"**{val}**"  # Or specific markup depending on implementation
    if self.italic:
      val = f"*{val}*"
    res += val
    res += "".join(t.text for t in self.trailing_trivia)
    return res
