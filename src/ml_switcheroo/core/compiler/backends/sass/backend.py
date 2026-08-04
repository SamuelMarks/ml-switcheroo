"""NVIDIA SASS Compiler Backend implementation for the ml-switcheroo project.

This module provides the `SassBackend` class, which is responsible for compiling
logical graphs of machine learning operations into low-level NVIDIA SASS
(Streaming Assembler) assembly code. It coordinates the synthesis of logical graph
structures into intermediate SASS AST structures, followed by the final text-based
emission.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
from ml_switcheroo.core.compiler.backends.sass.emitter import SassEmitter


class SassBackend(CompilerBackend):
  """Compiler Backend implementation for NVIDIA SASS.

  Orchestrates the synthesis (Graph -> AST) and emission (AST -> Text)
  for NVIDIA GPU Streaming Assembler targets.

  Attributes:
      synthesizer (SassSynthesizer): The synthesizer that translates LogicalGraph
          objects into SASS intermediate representations (AST).
      emitter (SassEmitter): The emitter that converts synthesized SASS nodes into
          runnable assembly text.
  """

  def __init__(self, semantics: Optional["SemanticsManager"] = None) -> None:
    """Initializes the SassBackend with necessary semantics management.

    Args:
        semantics (Optional[SemanticsManager]): A SemanticsManager instance to guide
            compilation and optimization constraints. If None, a default manager
            will be lazily instantiated.
    """
    # Lazy load if not provided, but typically passed from Registry/Engine
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    self.synthesizer = SassSynthesizer(semantics)
    self.emitter = SassEmitter()

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles LogicalGraph to SASS Assembly string.

    Args:
        graph (LogicalGraph): The intermediate representation of the logical computation graph
            to be compiled.

    Returns:
        str: The fully generated SASS assembly code representing the input logical graph.
    """
    sass_nodes = self.synthesizer.from_graph(graph)
    return self.emitter.emit(sass_nodes)
