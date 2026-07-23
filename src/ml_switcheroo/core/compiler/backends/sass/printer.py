"""SASS Printer (Visitor).

Provides the `SassPrinter` class which visits SASS AST nodes and
generates robustly formatted textual assembly output.
"""

from typing import List

from ml_switcheroo.core.compiler.frontends.sass.nodes import (
  Comment,
  Directive,
  Instruction,
  Label,
  SassNode,
)


class SassPrinter:
  """Prints SASS AST nodes into structured assembly strings using the Visitor pattern.

  Ensures consistent indentation and layout rules across all output.
  """

  def emit(self, nodes: List[SassNode]) -> str:
    """Emits the full SASS text for a sequence of nodes.

    Args:
        nodes: A list of SASS AST nodes.

    Returns:
        str: Formatted SASS assembly text.
    """
    lines = []
    for node in nodes:
      lines.append(self._visit(node))
    return "\n".join(lines) + "\n"

  def _visit(self, node: SassNode) -> str:
    """Dispatches to the correct visitor method."""
    if isinstance(node, Label):
      return self.visit_Label(node)
    elif isinstance(node, Instruction):
      return self.visit_Instruction(node)
    elif isinstance(node, Directive):
      return self.visit_Directive(node)
    elif isinstance(node, Comment):
      return self.visit_Comment(node)
    else:
      return self.visit_Fallback(node)

  def visit_Label(self, node: Label) -> str:
    """Visits a Label node (flush left)."""
    return str(node)

  def visit_Instruction(self, node: Instruction) -> str:
    """Visits an Instruction node (indented)."""
    return f"    {str(node)}"

  def visit_Directive(self, node: Directive) -> str:
    """Visits a Directive node (indented)."""
    return f"    {str(node)}"

  def visit_Comment(self, node: Comment) -> str:
    """Visits a Comment node (indented)."""
    return f"    {str(node)}"

  def visit_Fallback(self, node: SassNode) -> str:
    """Fallback handler for generic or custom nodes."""
    return f"    {str(node)}"
