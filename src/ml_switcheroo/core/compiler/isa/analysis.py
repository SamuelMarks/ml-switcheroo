"""Unified ISA Instruction Analyzer.

Provides a declarative, data-driven base analyzer for extracting high-level
semantic parameters (such as loop bounds, kernel sizes, feature dimensions,
and reduction element counts) from low-level hardware ISA basic blocks.
"""

import typing

from ml_switcheroo.core.compiler.isa.cst import IsaImmediate, IsaInstruction


class IsaAnalyzer:
  """Base analyzer for reverse-engineering high-level op metadata from ISA instructions."""

  CMP_OPCODE: str = ""
  IMMEDIATE_TYPE: typing.Type[IsaImmediate] = IsaImmediate
  LOOP_PARAMETER_RULES: typing.Mapping[str, typing.Tuple[str, str]] = {}

  @classmethod
  def analyze_block(
    cls,
    kind: str,
    instructions: typing.Sequence[IsaInstruction],
  ) -> typing.Dict[str, typing.Union[str, int, float]]:
    """Extract metadata from a block of instructions based on the operation kind.

    Args:
        kind: The operation type (e.g. ``"Conv2d"``, ``"Linear"``).
        instructions: Sequence of ISA instruction nodes inside the basic block.

    Returns:
        Dictionary of extracted parameters (e.g., ``{"kernel_size": 3, "arg_2": 3}``).
    """
    metadata: typing.Dict[str, typing.Union[str, int, float]] = {}
    loop_limits: typing.List[typing.Union[int, float]] = []

    for inst in instructions:
      if getattr(inst, "opcode", None) == cls.CMP_OPCODE:
        for op in getattr(inst, "operands", ()):
          if isinstance(op, cls.IMMEDIATE_TYPE):
            loop_limits.append(op.value)

    if not loop_limits:
      return metadata

    rule = cls.LOOP_PARAMETER_RULES.get(kind)
    if rule is not None:
      named_key, positional_key = rule
      limit_val = max(loop_limits)
      metadata[named_key] = limit_val
      metadata[positional_key] = limit_val

    return metadata
