"""Common Markup Concrete Syntax Tree (CST) Nodes for Visual DSLs.

Defines the shared AST node hierarchy for HTML, LaTeX, and TikZ visualizations.
"""

from dataclasses import dataclass, field
from typing import List

from ml_switcheroo.core.cst.base import CSTNode, Trivia


class MarkupNode:
  """Abstract base class for all markup DSL elements."""

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


@dataclass
class TriviaNode(CSTNode, MarkupNode):
  """Represent non-semantic textual elements (whitespace, newlines, comments)."""

  content: str = ""
  kind: str = "whitespace"
  leading_trivia: List[Trivia] = field(default_factory=list)
  trailing_trivia: List[Trivia] = field(default_factory=list)

  def emit(self, indent_level: int = 0) -> str:
    """Emit raw trivia content without modification.

    Args:
        indent_level: Current indentation depth (unused for trivia).

    Returns:
        str: Raw trivia text content.
    """
    return self.content


@dataclass
class TextNode(CSTNode, MarkupNode):
  """Represent standard text content within markup."""

  content: str = ""
  bold: bool = False
  italic: bool = False
  leading_trivia: List[Trivia] = field(default_factory=list)
  trailing_trivia: List[Trivia] = field(default_factory=list)

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
