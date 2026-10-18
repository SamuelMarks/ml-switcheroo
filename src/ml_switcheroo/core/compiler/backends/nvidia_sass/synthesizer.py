"""NVIDIA_SASS Synthesizer.

This module provides the "Middle-End" logic for the NVIDIA_SASS compiler pipeline.
"""

import os
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

import libcst as cst

import ml_switcheroo.core.compiler.backends.nvidia_sass.macros as sass_macros
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassOperand,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator
from ml_switcheroo.core.compiler.isa.synthesizer import IsaSynthesizer

if TYPE_CHECKING:
  pass

MAX_REGISTERS = 255


class NvidiaSassSynthesizer(IsaSynthesizer):
  """NVIDIA SASS Synthesizer inherited from unified base."""

  TARGET_NAME: str = "nvidia_sass"
  COMMENT_CLS: Type[NvidiaSassComment] = NvidiaSassComment
  INSTRUCTION_CLS: Type[NvidiaSassInstruction] = NvidiaSassInstruction
  LABEL_CLS: Type[NvidiaSassLabel] = NvidiaSassLabel
  IMMEDIATE_CLS: Type[NvidiaSassImmediate] = NvidiaSassImmediate
  LABEL_COMMENT_PREFIX: str = "NvidiaSassLabel"
  EMIT_BEGIN_END_COMMENTS_IN_PY: bool = True

  def __init__(self, semantics: Any = None) -> None:
    """Initialize the NVIDIA SASS synthesizer with a 255-register allocator and SASS macros.

    Args:
        semantics: SemanticsManager instance used to resolve operation variants.
    """
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    def reg_factory(idx: int) -> NvidiaSassRegister:
      """Construct an NvidiaSassRegister (`R{idx}`) for a physical register index.

      Args:
          idx: Zero-based physical register index.

      Returns:
          Constructed NvidiaSassRegister CST node.
      """
      return NvidiaSassRegister(name=f"R{idx}")

    allocator = GenericRegisterAllocator(MAX_REGISTERS, reg_factory)

    super().__init__(
      semantics=semantics,
      allocator=allocator,
      macros_yaml_path=os.path.join(os.path.dirname(__file__), "macros.yaml"),
      macros_json_path=os.path.join(os.path.dirname(__file__), "macros.json"),
      macro_module=sass_macros,
    )

  def _convert_instruction_to_py(self, inst: NvidiaSassInstruction) -> cst.SimpleStatementLine:
    """Convert a single NvidiaSassInstruction into a Python assignment or expression statement.

    Args:
        inst: The NvidiaSassInstruction CST node to convert.

    Returns:
        LibCST SimpleStatementLine invoking `nvidia_sass.<opcode>(...)`.
    """
    if not inst.operands:
      call = self._make_call(inst.opcode, [])
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    dest: Optional[Union[NvidiaSassRegister, NvidiaSassOperand]] = None
    srcs: List[NvidiaSassOperand] = []

    is_store = inst.opcode.startswith("ST")
    is_branch = inst.opcode in ["BRA", "BRX", "EXIT", "RET"]
    is_nop = inst.opcode == "NOP"

    if is_store or is_branch or is_nop:
      srcs = inst.operands
    else:
      dest = inst.operands[0]
      srcs = inst.operands[1:]

    arg_nodes = [cst.Arg(value=self._convert_operand_to_py(op)) for op in srcs]
    if inst.predicate:
      arg_nodes.append(cst.Arg(keyword=cst.Name("predicate"), value=cst.SimpleString(f"'{inst.predicate}'")))

    call = self._make_call(inst.opcode, arg_nodes)

    if dest:
      target_name = dest.name if isinstance(dest, NvidiaSassRegister) else str(dest)
      if not target_name.isidentifier():
        return cst.SimpleStatementLine(body=[cst.Expr(value=call)])
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(target_name))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: NvidiaSassOperand) -> cst.BaseExpression:
    """Convert an NvidiaSassOperand into a Python LibCST expression node.

    Args:
        op: The NvidiaSassOperand to convert.

    Returns:
        Corresponding LibCST Integer, Float, Name, or SimpleString expression.
    """
    if isinstance(op, NvidiaSassImmediate):
      return super()._convert_operand_to_py(op)

    raw = str(op)
    if raw.isalnum() and not raw.isdigit():
      return cst.Name(raw)
    return cst.SimpleString(f"'{raw}'")
