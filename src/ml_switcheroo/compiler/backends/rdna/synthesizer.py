"""
RDNA Synthesizer and Registry Backend.

This module provides the "Middle-End" logic for the RDNA compiler pipeline.
It bridges the gap between high-level Abstract Logic (LogicalGraphs)
and low-level Physical Assembly (Instruction nodes/Registers).

It contains:
1.  **RegisterAllocator**: A dual-pool allocator managing Scalar (SGPR) and
    Vector (VGPR) register files independently.
2.  **RdnaSynthesizer**:
    -   **Target Transformation (`from_graph`)**: Converts topological logical graphs
        into a linear list of RDNA instructions.
    -   **Source Transformation (`to_python`)**: Converts RDNA AST nodes back into
        Python LibCST nodes for high-level analysis or documentation.
3.  **RdnaBackend**: The CompilerBackend adapter for the Registry, including header generation.
"""

from typing import Dict, List, Optional, Union, Callable, TYPE_CHECKING
import libcst as cst

# Direct Import from Frontend to avoid circular dependency via core shims
from ml_switcheroo.compiler.frontends.rdna.nodes import (
  Instruction,
  SGPR,
  VGPR,
  RdnaNode,
  Comment,
  Operand,
  Label,
  Immediate,
)
from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.compiler.ir import LogicalGraph, topological_sort
from ml_switcheroo.compiler.backends.rdna.macros import expand_conv2d, expand_linear

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

# Physical limits based on RDNA architecture (simplified)
MAX_VGPR = 256
MAX_SGPR = 106


class RegisterAllocator:
  """
  Manages the mapping between symbolic variable names and physical registers.
  Maintains separate accounting for Scalar (SGPR) and Vector (VGPR) files.
  """

  def __init__(self) -> None:
    """Initializes the allocator with empty maps and counters."""
    self._var_to_vgpr: Dict[str, int] = {}
    self._var_to_sgpr: Dict[str, int] = {}
    self._next_vgpr = 0
    self._next_sgpr = 0

  def get_vector_register(self, var_name: str) -> VGPR:
    """
    Retrieves or allocates a Vector register (VGPR) for a symbolic variable.
    """
    if var_name in self._var_to_vgpr:
      return VGPR(self._var_to_vgpr[var_name])

    if self._next_vgpr >= MAX_VGPR:
      raise ValueError(f"VGPR overflow! Exceeded {MAX_VGPR} registers.")

    idx = self._next_vgpr
    self._var_to_vgpr[var_name] = idx
    self._next_vgpr += 1
    return VGPR(idx)

  def get_scalar_register(self, var_name: str) -> SGPR:
    """
    Retrieves or allocates a Scalar register (SGPR) for a symbolic variable.
    """
    if var_name in self._var_to_sgpr:
      return SGPR(self._var_to_sgpr[var_name])

    if self._next_sgpr >= MAX_SGPR:
      raise ValueError(f"SGPR overflow! Exceeded {MAX_SGPR} registers.")

    idx = self._next_sgpr
    self._var_to_sgpr[var_name] = idx
    self._next_sgpr += 1
    return SGPR(idx)

  def allocate_vector_temp(self) -> VGPR:
    """Allocates an anonymous temporary VGPR."""
    name = f"__v_temp_{self._next_vgpr}__"
    return self.get_vector_register(name)

  def allocate_scalar_temp(self) -> SGPR:
    """Allocates an anonymous temporary SGPR."""
    name = f"__s_temp_{self._next_sgpr}__"
    return self.get_scalar_register(name)

  def reset(self) -> None:
    """Resets all allocation state."""
    self._var_to_vgpr.clear()
    self._var_to_sgpr.clear()
    self._next_vgpr = 0
    self._next_sgpr = 0


class RdnaSynthesizer:
  """
  Bidirectional transpiler component for RDNA ISA.
  """

  def __init__(self, semantics: "SemanticsManager") -> None:
    self.semantics = semantics
    self.allocator = RegisterAllocator()
    self.macro_registry: Dict[str, Callable] = {
      "Conv2d": expand_conv2d,
      "Linear": expand_linear,
    }

  def from_graph(self, graph: LogicalGraph) -> List[RdnaNode]:
    """
    Converts a LogicalGraph into a list of RDNA AST nodes.
    """
    self.allocator.reset()
    output_nodes: List[RdnaNode] = []

    sorted_nodes = topological_sort(graph)

    # Build adjacency map: target_id -> [source_ids]
    input_map: Dict[str, List[str]] = {}
    for edge in graph.edges:
      if edge.target not in input_map:
        input_map[edge.target] = []
      input_map[edge.target].append(edge.source)

    for node in sorted_nodes:
      # --- Inputs ---
      if node.kind == "Input":
        reg = self.allocator.get_vector_register(node.id)
        var_name = node.metadata.get("name", node.id)
        output_nodes.append(Comment(f"Input {var_name} -> {reg}"))
        continue

      # --- Outputs ---
      if node.kind == "Output":
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_vector_register(sources[0])
          output_nodes.append(Comment(f"Return: {src_reg}"))
        continue

      # Resolve Abstract ID
      defn = self.semantics.get_definition(node.kind)
      abstract_id = defn[0] if defn else node.kind

      # --- Macro Expansion ---
      if abstract_id in self.macro_registry:
        expander = self.macro_registry[abstract_id]
        kernel_nodes = expander(self.allocator, node.id, node.metadata)
        output_nodes.extend(kernel_nodes)
        continue

      # --- 1:1 Instruction Synthesis ---
      variant = None
      if abstract_id:
        variant = self.semantics.resolve_variant(abstract_id, "rdna")

      if not variant or not variant.get("api"):
        output_nodes.append(Comment(f"Unmapped Op: {node.kind} ({node.id})"))
        continue

      opcode = variant["api"]

      # RDNA Vector ALU Format: OPCODE DST, SRC0, SRC1
      dst_reg = self.allocator.get_vector_register(node.id)
      operands: List[Operand] = [dst_reg]
      sources = input_map.get(node.id, [])

      for src_id in sources:
        # Assume inputs are in VGPRs for ALU ops
        src_reg = self.allocator.get_vector_register(src_id)
        operands.append(src_reg)

      inst = Instruction(opcode=opcode, operands=operands)
      output_nodes.append(inst)

    return output_nodes

  def to_python(self, rdna_nodes: List[RdnaNode]) -> cst.Module:
    """
    Converts RDNA AST nodes into a Python source structure representation.
    """
    body_stmts = []

    for node in rdna_nodes:
      stmt = None
      if isinstance(node, Instruction):
        stmt = self._convert_instruction_to_py(node)
      elif isinstance(node, Label):
        stmt = cst.SimpleStatementLine(
          body=[cst.Pass()],
          trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(f"# Label: {node.name}")),
        )

      if stmt:
        body_stmts.append(stmt)

    return cst.Module(body=body_stmts)

  def _convert_instruction_to_py(self, inst: Instruction) -> cst.SimpleStatementLine:
    if not inst.operands:
      call = self._make_call(inst.opcode, [])
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    # Heuristic: First operand is destination if it is a register and not a store/branch op
    is_store = "store" in inst.opcode
    is_branch = "branch" in inst.opcode

    dest: Optional[Operand] = None
    srcs: List[Operand] = []

    if is_store or is_branch:
      srcs = inst.operands
    else:
      # Standard ALU ops
      dest = inst.operands[0]
      srcs = inst.operands[1:]

    arg_nodes = []
    for op in srcs:
      val_node = self._convert_operand_to_py(op)
      arg_nodes.append(cst.Arg(value=val_node))

    call = self._make_call(inst.opcode, arg_nodes)

    if dest and isinstance(dest, (VGPR, SGPR)):
      target_name = str(dest)
      # Sanitize brackets for variable names
      clean_target = target_name.replace("[", "_").replace("]", "").replace(":", "_")
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(clean_target))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    else:
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: Operand) -> cst.BaseExpression:
    if isinstance(op, Immediate):
      if op.is_hex:
        return cst.Integer(hex(int(op.value)))
      if isinstance(op.value, float):
        return cst.Float(str(op.value))
      return cst.Integer(str(int(op.value)))

    raw = str(op)
    if "[" in raw:
      clean = raw.replace("[", "_").replace("]", "").replace(":", "_")
      return cst.Name(clean)

    if raw.isalnum() or "_" in raw:
      return cst.Name(raw)

    return cst.SimpleString(f"'{raw}'")

  def _make_call(self, opcode: str, args: List[cst.Arg]) -> cst.Call:
    return cst.Call(func=cst.Attribute(value=cst.Name("rdna"), attr=cst.Name(opcode)), args=args)


class RdnaBackend(CompilerBackend):
  """
  Compiler Backend implementation for AMD RDNA.
  Orchestrates the synthesis (Graph -> AST) and emission (AST -> Text).
  """

  def __init__(self, semantics: Optional["SemanticsManager"] = None) -> None:
    # Lazy load if not provided, but typically passed from Registry/Engine
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    self.synthesizer = RdnaSynthesizer(semantics)
    self.emitter = RdnaEmitter()
    # Default architecture for header generation matching legacy adapter defaults
    self.target_arch = "gfx1030"

  def compile(self, graph: LogicalGraph) -> str:
    """
    Compiles LogicalGraph to RDNA Assembly string.

    Args:
        graph: The intermediate representation.

    Returns:
        str: The RDNA code.
    """
    rdna_nodes = self.synthesizer.from_graph(graph)
    body = self.emitter.emit(rdna_nodes)
    header = f"; RDNA Code Generation Initialized (Arch: {self.target_arch})\n"
    return header + body
