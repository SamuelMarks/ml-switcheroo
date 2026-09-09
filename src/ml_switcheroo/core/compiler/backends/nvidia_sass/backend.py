"""NVIDIA SASS Compiler Backend implementation for the ml-switcheroo project.

This module provides the `NvidiaSassBackend` class, which is responsible for compiling
logical graphs of machine learning operations into low-level NVIDIA SASS
(Streaming Assembler) assembly code. It coordinates the synthesis of logical graph
structures into intermediate NVIDIA_SASS AST structures, followed by the final text-based
emission.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.backends.nvidia_sass.emitter import NvidiaSassEmitter


class NvidiaSassBackend(CompilerBackend):
  """Compiler Backend implementation for NVIDIA SASS.

  Orchestrates the synthesis (Graph -> AST) and emission (AST -> Text)
  for NVIDIA GPU Streaming Assembler targets.

  Attributes:
      synthesizer (NvidiaSassSynthesizer): The synthesizer that translates LogicalGraph
          objects into NVIDIA_SASS intermediate representations (AST).
      emitter (NvidiaSassEmitter): The emitter that converts synthesized NVIDIA_SASS nodes into
          runnable assembly text.
  """

  def __init__(self, semantics: Optional["SemanticsManager"] = None) -> None:
    """Initialize the NvidiaSassBackend with necessary semantics management.

    Args:
        semantics (Optional[SemanticsManager]): A SemanticsManager instance to guide
            compilation and optimization constraints. If None, a default manager
            will be lazily instantiated.
    """
    # Lazy load if not provided, but typically passed from Registry/Engine
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    self.synthesizer = NvidiaSassSynthesizer(semantics)
    self.emitter = NvidiaSassEmitter()

  def compile(self, graph: LogicalGraph) -> str:
    """Compile LogicalGraph to NVIDIA_SASS Assembly string.

    Args:
        graph (LogicalGraph): The intermediate representation of the logical computation graph
            to be compiled.

    Returns:
        str: The fully generated NVIDIA_SASS assembly code representing the input logical graph.
    """
    sass_nodes = self.synthesizer.from_graph(graph)
    return self.emitter.emit(sass_nodes)
