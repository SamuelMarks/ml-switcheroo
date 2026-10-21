"""NVIDIA_SASS Lifter (Frontend).

This module provides the logic to "lift" low-level NVIDIA_SASS AST nodes back into a
high-level `LogicalGraph`. It relies on semantic comment markers (e.g. `; BEGIN`)
to reconstruct the control flow and layer definitions.
"""

from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassInstruction,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.isa.analysis import IsaAnalyzer
from ml_switcheroo.core.compiler.isa.lifter import IsaLifter


class NvidiaSassLifter(IsaLifter[NvidiaSassNode, NvidiaSassInstruction, NvidiaSassComment]):
  """Reconstruct a LogicalGraph from a sequence of NVIDIA_SASS AST nodes."""

  DEFAULT_GRAPH_NAME: str = "DecompiledModel"
  COMMENT_TYPE: type[NvidiaSassComment] = NvidiaSassComment
  INSTRUCTION_TYPE: type[NvidiaSassInstruction] = NvidiaSassInstruction
  ANALYZER_CLS: type[IsaAnalyzer] = NvidiaSassAnalyzer
  ORPHAN_PREFIX: str = "asm"

  def _resolve_orphan_instruction(
    self,
    node: NvidiaSassInstruction,
    instruction_counter: int,
  ) -> tuple[str, str, dict[str, str | int | float]]:
    """Resolve an unenclosed NVIDIA SASS instruction into `(dest_name, kind, metadata)`.

    Args:
        node: The raw NvidiaSassInstruction node.
        instruction_counter: Monotonic counter for unnamed instructions.

    Returns:
        Tuple of `(dest_name, op_kind, metadata_dict)`.
    """
    dest_name = f"inst_{instruction_counter}"
    is_alu = node.opcode.upper() in {"FADD", "FMUL", "IADD3", "FFMA", "MOV"}
    if is_alu and node.operands and isinstance(node.operands[0], NvidiaSassRegister):
      dest_name = node.operands[0].name

    kind = f"asm.{node.opcode}"
    meta: dict[str, str | int | float] = {f"arg_{i}": str(op) for i, op in enumerate(node.operands)}
    return dest_name, kind, meta
