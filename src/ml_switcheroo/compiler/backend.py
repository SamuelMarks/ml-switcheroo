"""
Compiler Backend Protocol.

Defines the abstract interface for backends that consume the Logical Graph IR
and emit target specific code (e.g. RDNA Assembly, SASS, or High-Level Python).
"""

from abc import ABC, abstractmethod
from typing import Any
from ml_switcheroo.compiler.ir import LogicalGraph


class CompilerBackend(ABC):
  """
  Abstract base class for compilation backends.
  """

  @abstractmethod
  def compile(self, graph: LogicalGraph) -> Any:
    """
    Compiles the Logical Intermediate Representation (IR) into a target artifact.

    Args:
        graph (LogicalGraph): The intermediate representation of the model structure.

    Returns:
        Any: The compiled output (e.g., source code string, binary buffer, or AST).
    """
    pass
