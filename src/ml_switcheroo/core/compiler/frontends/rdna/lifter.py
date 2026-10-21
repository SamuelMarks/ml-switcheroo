"""RDNA Lifter (Frontend).

This module provides the logic to "lift" low-level RDNA AST nodes back into a
high-level `LogicalGraph`. It relies on semantic comment markers (e.g. `; BEGIN`)
emitted during compilation to reconstruct the control flow and layer definitions.
"""

from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaInstruction, RdnaNode
from ml_switcheroo.core.compiler.isa.analysis import IsaAnalyzer
from ml_switcheroo.core.compiler.isa.lifter import IsaLifter


class RdnaLifter(IsaLifter[RdnaNode, RdnaInstruction, RdnaComment]):
  """Reconstruct a LogicalGraph from a sequence of RDNA AST nodes."""

  DEFAULT_GRAPH_NAME: str = "DecompiledNet"
  COMMENT_TYPE: type[RdnaComment] = RdnaComment
  INSTRUCTION_TYPE: type[RdnaInstruction] = RdnaInstruction
  ANALYZER_CLS: type[IsaAnalyzer] = RdnaAnalyzer
  ORPHAN_PREFIX: str = "rdna"
