"""
RDNA Emitter (Backend).

Converts RDNA AST nodes into formatted assembly text.
"""

from typing import List
from ml_switcheroo.compiler.frontends.rdna.nodes import Label, RdnaNode


class RdnaEmitter:
  """
  Converts RDNA AST nodes into textual assembly code.
  """

  def emit(self, nodes: List[RdnaNode]) -> str:
    """
    Generates the RDNA source string from a list of nodes.
    """
    lines = []
    for node in nodes:
      prefix = "" if isinstance(node, Label) else "    "
      line = f"{prefix}{str(node)}"
      lines.append(line)

    return "\n".join(lines) + "\n"
