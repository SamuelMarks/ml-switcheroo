"""Doc."""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
from ml_switcheroo.core.compiler.backends.sass.emitter import SassEmitter


class SassBackend(CompilerBackend):
  """Compiler Backend implementation for NVIDIA SASS.
  Orchestrates the synthesis (Graph -> AST) and emission (AST -> Text).
  """

  def __init__(self, semantics: Optional["SemanticsManager"] = None) -> None:  # type: ignore
    """Execute implementation detail."""
    # Lazy load if not provided, but typically passed from Registry/Engine
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    self.synthesizer = SassSynthesizer(semantics)
    self.emitter = SassEmitter()

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles LogicalGraph to SASS Assembly string.

    Args:
        graph: The intermediate representation.

    Returns:
        str: The SASS code.

    """
    sass_nodes = self.synthesizer.from_graph(graph)
    return self.emitter.emit(sass_nodes)
