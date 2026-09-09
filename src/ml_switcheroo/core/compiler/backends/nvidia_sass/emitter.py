"""NVIDIA_SASS Emitter (Backend).

Converts NVIDIA_SASS AST nodes into formatted assembly text.
"""

from typing import List

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode
from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter


class NvidiaSassEmitter:
  """Convert NVIDIA_SASS AST nodes into textual assembly code."""

  def emit(self, nodes: List[NvidiaSassNode]) -> str:
    """Generate the NVIDIA_SASS source string from a list of nodes.

    Args:
        nodes (List[~ml_switcheroo.core.compiler.frontends.nvidia_sass.cst.NvidiaSassNode]): AST nodes.

    Returns:
        str: The formatted NVIDIA_SASS source code string.

    """
    printer = NvidiaSassPrinter()
    return printer.emit(nodes)
