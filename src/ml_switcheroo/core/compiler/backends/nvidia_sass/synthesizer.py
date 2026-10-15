"""NVIDIA_SASS Synthesizer.

This module provides the "Middle-End" logic for the NVIDIA_SASS compiler pipeline.
"""

import os
from typing import TYPE_CHECKING, List, Optional, Union

import libcst as cst

import ml_switcheroo.core.compiler.backends.nvidia_sass.macros as sass_macros
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassOperand,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, topological_sort
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator
from ml_switcheroo.core.compiler.isa.synthesizer import IsaSynthesizer

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

MAX_REGISTERS = 255


class NvidiaSassSynthesizer(IsaSynthesizer):
  """NVIDIA SASS Synthesizer inherited from unified base."""

  def __init__(self, semantics: "SemanticsManager") -> None:
    """Initialize the NVIDIA SASS synthesizer with a 255-register allocator and SASS macros.

    Args:
        semantics: SemanticsManager instance used to resolve operation variants.
    """

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

  def from_graph(self, graph: LogicalGraph) -> List[NvidiaSassNode]:
    """Synthesize a linear sequence of NVIDIA SASS CST nodes from a LogicalGraph.

    Args:
        graph: LogicalGraph intermediate representation to lower to SASS.

    Returns:
        Ordered list of synthesized NvidiaSassNode instructions, labels, and comments.
    """
    self.allocator.reset()
    self.allocator.build_liveness(graph)
    output_nodes: List[NvidiaSassNode] = []
    sorted_nodes = topological_sort(graph)
    input_map = self._get_input_map(graph)

    for node in sorted_nodes:
      op_type: str = str(node.op_type if hasattr(node, "op_type") else getattr(node, "kind", ""))
      node_attrs = getattr(node, "attributes", {})

      if op_type == "Input":
        reg = self.allocator.get_register(node.id)
        var_name = node_attrs.get("name", node.id)
        output_nodes.append(NvidiaSassComment(text=f"Input {var_name} -> {reg.name}"))

      elif op_type == "Output":
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_register(sources[0])
          output_nodes.append(NvidiaSassComment(text=f"Return: {src_reg.name}"))
          self.allocator.record_usage(sources[0])

      else:
        abstract_id = self._resolve_abstract_id(op_type)

        if abstract_id in self.macro_registry:
          expander = self.macro_registry[abstract_id]
          kernel_nodes = expander(self.allocator, node.id, node_attrs)
          output_nodes.extend(kernel_nodes)
          sources = input_map.get(node.id, [])
          for src_id in sources:
            self.allocator.record_usage(src_id)
          continue

        suffix_id = abstract_id.split(".")[-1] if abstract_id else ""
        if suffix_id and suffix_id in self.macro_registry:
          expander = self.macro_registry[suffix_id]
          kernel_nodes = expander(self.allocator, node.id, node_attrs)
          output_nodes.extend(kernel_nodes)
          sources = input_map.get(node.id, [])
          for src_id in sources:
            self.allocator.record_usage(src_id)
          continue

        variant = None
        if abstract_id:
          variant = self.semantics.resolve_variant(abstract_id, "nvidia_sass")

        if not variant or not variant.get("api"):
          output_nodes.append(NvidiaSassComment(text=f"Unmapped Op: {op_type} ({node.id})"))
          continue

        opcode = variant["api"]
        dst_reg = self.allocator.get_register(node.id)
        operands: List[NvidiaSassOperand] = [dst_reg]

        sources = input_map.get(node.id, [])
        for src_id in sources:
          src_reg = self.allocator.get_register(src_id)
          operands.append(src_reg)
          self.allocator.record_usage(src_id)

        inst = NvidiaSassInstruction(opcode=opcode, operands=operands)
        output_nodes.append(inst)

    return output_nodes

  def to_python(self, sass_nodes: List[NvidiaSassNode]) -> cst.Module:
    """Convert a list of NVIDIA SASS CST nodes into a high-level Python LibCST Module.

    Args:
        sass_nodes: Sequence of NvidiaSassNode instances to lift into Python statements.

    Returns:
        Constructed LibCST Module representing the SASS instruction stream.
    """
    body_stmts = []
    for node in sass_nodes:
      stmt = None
      if isinstance(node, NvidiaSassInstruction):
        stmt = self._convert_instruction_to_py(node)
      elif isinstance(node, NvidiaSassComment):
        if "BEGIN" in node.text or "END" in node.text:
          stmt = cst.SimpleStatementLine(
            body=[cst.Pass()],
            trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(value=f"# {node.text}")),
          )
      elif isinstance(node, NvidiaSassLabel):
        stmt = cst.SimpleStatementLine(
          body=[cst.Pass()],
          trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(value=f"# NvidiaSassLabel: {node.name}")),
        )

      if stmt:
        body_stmts.append(stmt)
    return cst.Module(body=body_stmts)

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

    arg_nodes = []
    for op in srcs:
      arg_val = self._convert_operand_to_py(op)
      arg_nodes.append(cst.Arg(value=arg_val))

    if inst.predicate:
      arg_nodes.append(cst.Arg(keyword=cst.Name("predicate"), value=cst.SimpleString(f"'{inst.predicate}'")))

    call = self._make_call(inst.opcode, arg_nodes)

    if dest:
      target_name = str(dest)
      if not target_name.isidentifier():
        return cst.SimpleStatementLine(body=[cst.Expr(value=call)])
      if isinstance(dest, NvidiaSassRegister):
        target_name = dest.name
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(target_name))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    else:
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: NvidiaSassOperand) -> cst.BaseExpression:
    """Convert an NvidiaSassOperand into a Python LibCST expression node.

    Args:
        op: The NvidiaSassOperand to convert.

    Returns:
        Corresponding LibCST Integer, Float, Name, or SimpleString expression.
    """
    if isinstance(op, NvidiaSassImmediate):
      if op.is_hex:
        return cst.Integer(hex(int(op.value)))
      if isinstance(op.value, float):
        return cst.Float(str(op.value))
      return cst.Integer(str(int(op.value)))

    raw = str(op)
    if raw.isalnum() and not raw.isdigit():
      return cst.Name(raw)
    return cst.SimpleString(f"'{raw}'")

  def _make_call(self, opcode: str, args: List[cst.Arg]) -> cst.Call:
    """Construct a `nvidia_sass.<opcode>(*args)` LibCST Call node.

    Args:
        opcode: NVIDIA SASS instruction mnemonic.
        args: List of LibCST Arg nodes for the instruction operands.

    Returns:
        Constructed LibCST Call node.
    """
    return cst.Call(func=cst.Attribute(value=cst.Name("nvidia_sass"), attr=cst.Name(opcode)), args=args)
