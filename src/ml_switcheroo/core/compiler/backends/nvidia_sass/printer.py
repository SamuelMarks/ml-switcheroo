"""NVIDIA_SASS Printer (Visitor).

Provides the `NvidiaSassPrinter` class which visits NVIDIA_SASS AST nodes and
generates robustly formatted textual assembly output.
"""

from typing import List

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
)


class NvidiaSassPrinter:
  """Print NVIDIA_SASS AST nodes into structured assembly strings using the Visitor pattern.

  Ensures consistent indentation and layout rules across all output.
  """

  def emit(self, nodes: List[NvidiaSassNode]) -> str:
    """Emit the full NVIDIA_SASS text for a sequence of nodes.

    Args:
        nodes: A list of NVIDIA_SASS AST nodes.

    Returns:
        str: Formatted NVIDIA_SASS assembly text.
    """
    lines = []
    for node in nodes:
      lines.append(self._visit(node))
    return "\n".join(lines) + "\n"

  def _visit(self, node: NvidiaSassNode) -> str:
    """Dispatch to the correct visitor method based on the node type.

    Args:
        node: The NVIDIA_SASS AST node to visit.

    Returns:
        str: Formatted NVIDIA_SASS assembly text of the node.
    """
    if isinstance(node, NvidiaSassLabel):
      return self.visit_Label(node)
    elif isinstance(node, NvidiaSassInstruction):
      return self.visit_Instruction(node)
    elif isinstance(node, NvidiaSassDirective):
      return self.visit_Directive(node)
    elif isinstance(node, NvidiaSassComment):
      return self.visit_Comment(node)
    else:
      return self.visit_Fallback(node)

  def visit_Label(self, node: NvidiaSassLabel) -> str:
    """Visit a NvidiaSassLabel node (flush left).

    Args:
        node: The NvidiaSassLabel node to visit.

    Returns:
        str: Formatted NVIDIA_SASS label assembly text.
    """
    return str(node)

  def visit_Instruction(self, node: NvidiaSassInstruction) -> str:
    """Visit an NvidiaSassInstruction node (indented).

    Args:
        node: The NvidiaSassInstruction node to visit.

    Returns:
        str: Formatted NVIDIA_SASS instruction assembly text with indentation.
    """
    return f"    {str(node)}"

  def visit_Directive(self, node: NvidiaSassDirective) -> str:
    """Visit a NvidiaSassDirective node (indented).

    Args:
        node: The NvidiaSassDirective node to visit.

    Returns:
        str: Formatted NVIDIA_SASS directive assembly text with indentation.
    """
    return f"    {str(node)}"

  def visit_Comment(self, node: NvidiaSassComment) -> str:
    """Visit a NvidiaSassComment node (indented).

    Args:
        node: The NvidiaSassComment node to visit.

    Returns:
        str: Formatted NVIDIA_SASS comment assembly text with indentation.
    """
    return f"    {str(node)}"

  def visit_Fallback(self, node: NvidiaSassNode) -> str:
    """Fallback handler for generic or custom nodes.

    Args:
        node: The custom or generic NvidiaSassNode to visit.

    Returns:
        str: Formatted generic node assembly text with indentation.
    """
    return f"    {str(node)}"
