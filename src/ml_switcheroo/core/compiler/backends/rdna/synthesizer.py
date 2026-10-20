"""RDNA Synthesizer and Registry Backend.

This module provides the "Middle-End" logic for the RDNA compiler pipeline.
It bridges the gap between high-level Abstract Logic (LogicalGraphs)
and low-level Physical Assembly (RdnaInstruction nodes/Registers).
"""

import os
import typing

import libcst as cst
from ml_switcheroo_ir import AttributeValue

import ml_switcheroo.core.compiler.backends.rdna.macros as rdna_macros
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator
from ml_switcheroo.core.compiler.isa.cst import IsaInstruction, IsaNode
from ml_switcheroo.core.compiler.isa.synthesizer import IsaSynthesizer
from ml_switcheroo.core.rewriter.context import SemanticsProtocol

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

MAX_VGPR = 256
MAX_SGPR = 106


class RdnaSynthesizer(IsaSynthesizer[RdnaNode, RdnaVGPR]):
  """Bidirectional transpiler component for RDNA ISA."""

  TARGET_NAME: str = "rdna"
  COMMENT_CLS: typing.Type[RdnaComment] = RdnaComment
  INSTRUCTION_CLS: typing.Type[RdnaInstruction] = RdnaInstruction
  LABEL_CLS: typing.Type[RdnaLabel] = RdnaLabel
  IMMEDIATE_CLS: typing.Type[RdnaImmediate] = RdnaImmediate
  LABEL_COMMENT_PREFIX: str = "RdnaLabel"
  EMIT_BEGIN_END_COMMENTS_IN_PY: bool = False

  def __init__(self, semantics: typing.Optional[typing.Union["SemanticsManager", SemanticsProtocol]] = None) -> None:
    """Initialize the RDNA synthesizer with a 256-VGPR register allocator and RDNA macros.

    Args:
        semantics: SemanticsManager instance used to resolve RDNA operation variants.
    """
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    def reg_factory(idx: int) -> RdnaVGPR:
      """Construct an RdnaVGPR (`v{idx}`) for a physical vector register index.

      Args:
          idx: Zero-based VGPR register index.

      Returns:
          Constructed RdnaVGPR CST node.
      """
      return RdnaVGPR(index=idx)

    allocator = GenericRegisterAllocator(MAX_VGPR, reg_factory)

    super().__init__(
      semantics=semantics,
      allocator=allocator,
      macros_yaml_path=os.path.join(os.path.dirname(__file__), "macros.yaml"),
      macros_json_path=os.path.join(os.path.dirname(__file__), "macros.json"),
      macro_module=rdna_macros,
    )

  def _handle_custom_opcode(
    self,
    opcode: str,
    node_id: str,
    node_attrs: typing.Mapping[str, AttributeValue],
  ) -> typing.Optional[typing.List[RdnaNode]]:
    """Handle RDNA `; Macro.<Name>` directives and comment-prefixed opcodes.

    Args:
        opcode: Resolved RDNA opcode string.
        node_id: Logical node identifier.
        node_attrs: Logical node attributes dictionary.

    Returns:
        List of emitted RdnaNode instances if handled, or ``None`` for standard ALU instructions.
    """
    clean_macro: typing.Optional[str] = None
    if opcode.startswith("; Macro."):
      clean_macro = opcode[len("; Macro.") :].strip()
    elif opcode.startswith("Macro."):
      clean_macro = opcode[len("Macro.") :].strip()

    if clean_macro:
      if clean_macro in self.macro_registry:
        return list(self.macro_registry[clean_macro](self.allocator, node_id, node_attrs))
      return [
        RdnaComment(text=f"BEGIN {clean_macro} ({node_id})"),
        RdnaComment(text=f"END {clean_macro} ({node_id})"),
      ]

    if opcode.startswith(";"):
      return [RdnaComment(text=f"Unmapped Op: {opcode} ({node_id})")]

    return None

  def _convert_instruction_to_py(self, inst: IsaInstruction) -> cst.SimpleStatementLine:
    """Convert RDNA instruction to Python AST.

    Args:
        inst: The RDNA Instruction.

    Returns:
        A CST statement.
    """
    if not inst.operands:
      call = self._make_call(inst.opcode, [])
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    is_store = "store" in inst.opcode
    is_branch = "branch" in inst.opcode

    dest: typing.Optional[IsaNode] = None
    srcs: typing.List[IsaNode] = []

    if is_store or is_branch:
      srcs = list(inst.operands)
    else:
      dest = inst.operands[0]
      srcs = list(inst.operands[1:])

    arg_nodes = [cst.Arg(value=self._convert_operand_to_py(op)) for op in srcs]
    call = self._make_call(inst.opcode, arg_nodes)

    if dest and isinstance(dest, (RdnaVGPR, RdnaSGPR)):
      target_name = str(dest)
      clean_target = target_name.replace("[", "_").replace("]", "").replace(":", "_")
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(clean_target))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    return cst.SimpleStatementLine(body=[cst.Expr(value=call)])


class RdnaBackend(CompilerBackend):
  """Compiler Backend implementation for AMD RDNA.

  Orchestrates the synthesis (Graph -> AST) and emission (AST -> Text).
  """

  def __init__(self, semantics: typing.Optional["SemanticsManager"] = None) -> None:
    """Initialize RDNA Backend.

    Args:
        semantics: Optional SemanticsManager.
    """
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    self.synthesizer = RdnaSynthesizer(semantics)
    self.emitter = RdnaEmitter()
    self.target_arch = "gfx1030"

  def compile(self, graph: LogicalGraph) -> str:
    """Compile LogicalGraph to RDNA Assembly string.

    Args:
        graph: The intermediate representation.

    Returns:
        str: The RDNA code.
    """
    rdna_nodes = self.synthesizer.from_graph(graph)
    body = self.emitter.emit(rdna_nodes)
    header = f"; RDNA Code Generation Initialized (Arch: {self.target_arch})\n"
    return header + body
