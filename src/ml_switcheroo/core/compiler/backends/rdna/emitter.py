"""RDNA Emitter (Backend).

Converts RDNA AST nodes into formatted assembly text.
"""

from typing import List
from ml_switcheroo.core.compiler.frontends.rdna.nodes import RdnaNode
from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter


class RdnaEmitter:
  """Converts RDNA AST nodes into textual assembly code."""

  def emit(self, nodes: List[RdnaNode]) -> str:
    """Generates the RDNA source string from a list of nodes."""
    printer = RdnaPrinter()
    return printer.emit(nodes)
