"""
RDNA Emitter.

This module provides the `RdnaEmitter` class, which converts the Abstract Syntax Tree (AST)
nodes (Instructions, Labels, Comments) into formatted text valid for AMD GCN/RDNA assemblers.

Capabilities:
1.  **Indentation**: Labels are flush-left; Instructions and Directives are indented.
2.  **Formatting**: Delegates string conversion to the specific node's `__str__` method.
"""

from typing import List

from ml_switcheroo.core.rdna.nodes import Label, RdnaNode


class RdnaEmitter:
  """
  Converts RDNA AST nodes into textual assembly code.
  """

  def emit(self, nodes: List[RdnaNode]) -> str:
    """
    Generates the RDNA source string from a list of nodes.

    Formatting Rules:
    - Labels (e.g. `L_1:`) are rendered flush-left.
    - All other nodes (Instructions, Comments, Directives) are indented by 4 spaces.
    - Each node occupies one line.

    Args:
        nodes (List[RdnaNode]): A list of RDNA AST nodes.

    Returns:
        str: The formatted RDNA source code string.
    """
    lines = []
    for node in nodes:
      # Labels flush left, everything else indented
      prefix = "" if isinstance(node, Label) else "    "
      line = f"{prefix}{str(node)}"
      lines.append(line)

    # Ensure file ends with newline
    return "\n".join(lines) + "\n"
