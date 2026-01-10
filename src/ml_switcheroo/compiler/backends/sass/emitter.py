"""
SASS Emitter (Backend).

Converts SASS AST nodes into formatted assembly text.
"""

from typing import List

from ml_switcheroo.compiler.frontends.sass.nodes import Label, SassNode


class SassEmitter:
  """
  Converts SASS AST nodes into textual assembly code.
  """

  def emit(self, nodes: List[SassNode]) -> str:
    """
    Generates the SASS source string from a list of nodes.

    Formatting Rules:
    - Labels (e.g. `L_1:`) are rendered flush-left.
    - All other nodes are indented by 4 spaces.

    Args:
        nodes (List[SassNode]): AST nodes.

    Returns:
        str: The formatted SASS source code string.
    """
    lines = []
    for node in nodes:
      prefix = "" if isinstance(node, Label) else "    "
      line = f"{prefix}{str(node)}"
      lines.append(line)

    return "\n".join(lines) + "\n"
