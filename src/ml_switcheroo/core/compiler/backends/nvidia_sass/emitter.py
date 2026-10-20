"""NVIDIA_SASS Emitter (Backend).

Converts NVIDIA_SASS AST nodes into formatted assembly text.
"""

import typing

from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter
from ml_switcheroo.core.compiler.isa.cst import IsaNode


class NvidiaSassEmitter:
  """Convert NVIDIA_SASS AST nodes into textual assembly code."""

  def emit(self, nodes: typing.Sequence[IsaNode]) -> str:
    """Generate the NVIDIA_SASS source string from a list of nodes.

    Args:
        nodes (Sequence[~ml_switcheroo.core.compiler.isa.cst.IsaNode]): AST nodes.

    Returns:
        str: The formatted NVIDIA_SASS source code string.

    """
    printer = NvidiaSassPrinter()
    return printer.emit(nodes)
