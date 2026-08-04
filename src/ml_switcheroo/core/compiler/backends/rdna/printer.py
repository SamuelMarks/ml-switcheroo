"""RDNA Printer (Visitor).

Provides the `RdnaPrinter` class which visits RDNA AST nodes and
generates robustly formatted textual assembly output.
"""

from typing import List

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaInstruction,
  RdnaLabel,
  RdnaNode,
)


class RdnaPrinter:
  """Prints RDNA AST nodes into structured assembly strings using the Visitor pattern.

  Ensures consistent indentation and layout rules across all output.
  """

  def emit(self, nodes: List[RdnaNode]) -> str:
    """Emits the full RDNA text for a sequence of nodes.

    Args:
        nodes: A list of RDNA AST nodes.

    Returns:
        str: Formatted RDNA assembly text.
    """
    lines = []
    for node in nodes:
      lines.append(self._visit(node))
    return "\n".join(lines) + "\n"

  def _visit(self, node: RdnaNode) -> str:
    """Dispatches to the correct visitor method.

    Args:
        node: The RDNA node to visit.

    Returns:
        str: The formatted string representing the visited node.
    """
    if isinstance(node, RdnaLabel):
      return self.visit_Label(node)
    elif isinstance(node, RdnaInstruction):
      return self.visit_Instruction(node)
    elif isinstance(node, RdnaDirective):
      return self.visit_Directive(node)
    elif isinstance(node, RdnaComment):
      return self.visit_Comment(node)
    else:
      return self.visit_Fallback(node)

  def visit_Label(self, node: RdnaLabel) -> str:
    """Visits a RdnaLabel node (flush left).

    Args:
        node: The RDNA label node to process.

    Returns:
        str: The string representation of the label.
    """
    return str(node)

  def visit_Instruction(self, node: RdnaInstruction) -> str:
    """Visits an RdnaInstruction node (indented).

    Args:
        node: The RDNA instruction node to process.

    Returns:
        str: The indented string representation of the instruction.
    """
    return f"    {str(node)}"

  def visit_Directive(self, node: RdnaDirective) -> str:
    """Visits a RdnaDirective node (indented).

    Args:
        node: The RDNA directive node to process.

    Returns:
        str: The indented string representation of the directive.
    """
    return f"    {str(node)}"

  def visit_Comment(self, node: RdnaComment) -> str:
    """Visits a RdnaComment node (indented).

    Args:
        node: The RDNA comment node to process.

    Returns:
        str: The indented string representation of the comment.
    """
    return f"    {str(node)}"

  def visit_Fallback(self, node: RdnaNode) -> str:
    """Fallback handler for generic or custom nodes.

    Args:
        node: The generic RDNA node to process.

    Returns:
        str: The indented string representation of the fallback node.
    """
    return f"    {str(node)}"
