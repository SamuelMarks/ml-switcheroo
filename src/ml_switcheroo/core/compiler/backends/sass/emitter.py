"""SASS Emitter (Backend).

Converts SASS AST nodes into formatted assembly text.
"""

from typing import List

from ml_switcheroo.core.compiler.frontends.sass.cst import SassNode
from ml_switcheroo.core.compiler.backends.sass.printer import SassPrinter


class SassEmitter:
  """Converts SASS AST nodes into textual assembly code."""

  def emit(self, nodes: List[SassNode]) -> str:
    """Generates the SASS source string from a list of nodes.

    Args:
        nodes (List[~ml_switcheroo.core.compiler.frontends.sass.cst.SassNode]): AST nodes.

    Returns:
        str: The formatted SASS source code string.

    """
    printer = SassPrinter()
    return printer.emit(nodes)
