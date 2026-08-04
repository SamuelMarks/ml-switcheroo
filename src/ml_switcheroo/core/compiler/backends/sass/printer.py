"""SASS Printer (Visitor).

Provides the `SassPrinter` class which visits SASS AST nodes and
generates robustly formatted textual assembly output.
"""

from typing import List

from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassDirective,
  SassInstruction,
  SassLabel,
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
    """Dispatches to the correct visitor method based on the node type.

    Args:
        node: The SASS AST node to visit.

    Returns:
        str: Formatted SASS assembly text of the node.
    """
    if isinstance(node, SassLabel):
      return self.visit_Label(node)
    elif isinstance(node, SassInstruction):
      return self.visit_Instruction(node)
    elif isinstance(node, SassDirective):
      return self.visit_Directive(node)
    elif isinstance(node, SassComment):
      return self.visit_Comment(node)
    else:
      return self.visit_Fallback(node)

  def visit_Label(self, node: SassLabel) -> str:
    """Visits a SassLabel node (flush left).

    Args:
        node: The SassLabel node to visit.

    Returns:
        str: Formatted SASS label assembly text.
    """
    return str(node)

  def visit_Instruction(self, node: SassInstruction) -> str:
    """Visits an SassInstruction node (indented).

    Args:
        node: The SassInstruction node to visit.

    Returns:
        str: Formatted SASS instruction assembly text with indentation.
    """
    return f"    {str(node)}"

  def visit_Directive(self, node: SassDirective) -> str:
    """Visits a SassDirective node (indented).

    Args:
        node: The SassDirective node to visit.

    Returns:
        str: Formatted SASS directive assembly text with indentation.
    """
    return f"    {str(node)}"

  def visit_Comment(self, node: SassComment) -> str:
    """Visits a SassComment node (indented).

    Args:
        node: The SassComment node to visit.

    Returns:
        str: Formatted SASS comment assembly text with indentation.
    """
    return f"    {str(node)}"

  def visit_Fallback(self, node: SassNode) -> str:
    """Fallback handler for generic or custom nodes.

    Args:
        node: The custom or generic SassNode to visit.

    Returns:
        str: Formatted generic node assembly text with indentation.
    """
    return f"    {str(node)}"
