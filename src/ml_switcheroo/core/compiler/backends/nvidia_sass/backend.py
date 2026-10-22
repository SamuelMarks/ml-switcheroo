"""NVIDIA SASS Compiler Backend implementation for the ml-switcheroo project.

This module provides the `NvidiaSassBackend` class, which is responsible for compiling
logical graphs of machine learning operations into low-level NVIDIA SASS
(Streaming Assembler) assembly code. It coordinates the synthesis of logical graph
structures into intermediate NVIDIA_SASS AST structures, followed by the final text-based
emission.
"""

import typing

if typing.TYPE_CHECKING:
  from ml_ecosystem_snapshots.grounding.models import GroundingReport

  from ml_switcheroo.semantics.manager import SemanticsManager

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.nvidia_sass.emitter import NvidiaSassEmitter
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.ir import LogicalGraph


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

  def __init__(self, semantics: "SemanticsManager | None" = None) -> None:
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

  def validate_instruction(
    self,
    mnemonic: str,
    architecture: str = "sm_80",
    operands: list[str] | None = None,
    modifiers: list[str] | None = None,
  ) -> "GroundingReport | None":
    """Validate a SASS instruction mnemonic against grounded hardware specifications.

    Args:
        mnemonic (str): The instruction mnemonic (e.g. 'FADD', 'FFMA').
        architecture (str): Target GPU architecture (defaults to 'sm_80').
        operands (Optional[list[str]]): Optional register/operand tokens.
        modifiers (Optional[list[str]]): Optional instruction modifiers.

    Returns:
        Optional[GroundingReport]: A GroundingReport if ml_ecosystem_snapshots is available, or None.
    """
    try:
      from ml_ecosystem_snapshots.grounding.hardware import validate_sass_instruction

      report: "GroundingReport" = validate_sass_instruction(
        mnemonic=mnemonic,
        architecture=architecture,
        operands=operands or ["R0", "R1", "R2"],
        modifiers=modifiers,
      )
      return report
    except Exception:
      return None
