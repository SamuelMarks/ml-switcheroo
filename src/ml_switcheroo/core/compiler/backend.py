"""Compiler Backend Protocol.

Defines the abstract interface for backends that consume the Logical Graph IR
and emit target specific code (e.g. RDNA Assembly, NVIDIA_SASS, or High-Level Python).
"""

from abc import ABC, abstractmethod
import typing

from ml_switcheroo.core.compiler.ir import IrCompilerBackend, LogicalGraph

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class CompilerBackend(IrCompilerBackend[str], ABC):
  """Abstract base class for compilation backends."""

  def __init__(
    self,
    semantics: "SemanticsManager | None" = None,
    *,
    framework: str = "torch",
    format: str = "json",
  ) -> None:
    """Initialize the compiler backend with optional semantic knowledge and target options.

    Args:
        semantics: Optional semantics manager for operator and attribute resolution.
        framework: Target framework identifier for high-level synthesis backends.
        format: Serialization format ('json' or 'python') for IR backends.
    """
    self.semantics = semantics

  @abstractmethod
  def compile(self, graph: LogicalGraph) -> str:
    """Compile the Logical Intermediate Representation (IR) into a target artifact.

    Args:
        graph (LogicalGraph): The intermediate representation of the model structure.

    Returns:
        str: The compiled output (e.g., source code string, binary buffer, or AST).

    """
    raise NotImplementedError
