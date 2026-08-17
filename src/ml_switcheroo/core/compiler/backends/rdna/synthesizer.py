"""RDNA Synthesizer and Registry Backend.

This module provides the "Middle-End" logic for the RDNA compiler pipeline.
It bridges the gap between high-level Abstract Logic (LogicalGraphs)
and low-level Physical Assembly (RdnaInstruction nodes/Registers).

It contains:

1.  **RegisterAllocator**: A dual-pool allocator managing Scalar (RdnaSGPR) and
    Vector (RdnaVGPR) register files independently.
2.  **RdnaSynthesizer**:

    -   **Target Transformation (`from_graph`)**: Converts topological logical graphs
        into a linear list of RDNA instructions.
    -   **Source Transformation (`to_python`)**: Converts RDNA AST nodes back into
        Python LibCST nodes for high-level analysis or documentation.
3.  **RdnaBackend**: The CompilerBackend adapter for the Registry, including header generation.
"""

from typing import Any

from typing import Dict, List, Optional, Callable, TYPE_CHECKING
import libcst as cst

# Direct Import from Frontend to avoid circular dependency via core shims
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaInstruction,
  RdnaSGPR,
  RdnaVGPR,
  RdnaNode,
  RdnaComment,
  RdnaOperand,
  RdnaLabel,
  RdnaImmediate,
)
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.core.compiler.ir import LogicalGraph, topological_sort
import ml_switcheroo.core.compiler.backends.rdna.macros as rdna_macros
import json
import os

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

# Physical limits based on RDNA architecture (simplified)
MAX_VGPR = 256
MAX_SGPR = 106


class RegisterAllocator:
  """Manages the mapping between symbolic variable names and physical registers.

  Maintains separate accounting for Scalar (RdnaSGPR) and Vector (RdnaVGPR) files.
  """

  def __init__(self) -> None:
    """Initializes the allocator with empty maps and counters."""
    self._var_to_vgpr: Dict[str, int] = {}
    self._var_to_sgpr: Dict[str, int] = {}
    self._next_vgpr = 0
    self._next_sgpr = 0

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Retrieves or allocates a Vector register (RdnaVGPR) for a symbolic variable.

    Raises:
        ValueError: If register limit is exceeded.

    Args:
        var_name: The symbolic name of the variable.

    Returns:
        RdnaVGPR: The allocated or retrieved vector register.
    """
    if var_name in self._var_to_vgpr:
      return RdnaVGPR(index=self._var_to_vgpr[var_name])

    if self._next_vgpr >= MAX_VGPR:
      raise ValueError(f"RdnaVGPR overflow! Exceeded {MAX_VGPR} registers.")

    idx = self._next_vgpr
    self._var_to_vgpr[var_name] = idx
    self._next_vgpr += 1
    return RdnaVGPR(index=idx)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Retrieves or allocates a Scalar register (RdnaSGPR) for a symbolic variable.

    Raises:
        ValueError: If register limit is exceeded.

    Args:
        var_name: The symbolic name of the variable.

    Returns:
        RdnaSGPR: The allocated or retrieved scalar register.
    """
    if var_name in self._var_to_sgpr:
      return RdnaSGPR(index=self._var_to_sgpr[var_name])

    if self._next_sgpr >= MAX_SGPR:
      raise ValueError(f"RdnaSGPR overflow! Exceeded {MAX_SGPR} registers.")

    idx = self._next_sgpr
    self._var_to_sgpr[var_name] = idx
    self._next_sgpr += 1
    return RdnaSGPR(index=idx)

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Allocates an anonymous temporary RdnaVGPR.

    Returns:
        RdnaVGPR: The newly allocated temporary vector register.
    """
    name = f"__v_temp_{self._next_vgpr}__"
    return self.get_vector_register(name)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Allocates an anonymous temporary RdnaSGPR.

    Returns:
        RdnaSGPR: The newly allocated temporary scalar register.
    """
    name = f"__s_temp_{self._next_sgpr}__"
    return self.get_scalar_register(name)

  def reset(self) -> None:
    """Resets all allocation state."""
    self._var_to_vgpr.clear()
    self._var_to_sgpr.clear()
    self._next_vgpr = 0
    self._next_sgpr = 0


class RdnaSynthesizer:
  """Bidirectional transpiler component for RDNA ISA."""

  def __init__(self, semantics: "SemanticsManager") -> None:
    """Initialize RDNA Synthesizer.

    Args:
        semantics: SemanticsManager for opcode resolution.
    """
    self.semantics = semantics
    self.allocator = RegisterAllocator()
    self.macro_registry: Dict[str, Callable[..., Any]] = {}
    macros_json_path = os.path.join(os.path.dirname(__file__), "macros.json")
    if os.path.exists(macros_json_path):
      with open(macros_json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

      for key, func_name in mapping.items():
        if hasattr(rdna_macros, func_name):
          self.macro_registry[key] = getattr(rdna_macros, func_name)

  def from_graph(self, graph: LogicalGraph) -> List[RdnaNode]:
    """Converts a LogicalGraph into a list of RDNA AST nodes.

    Args:
        graph: The LogicalGraph representing the operations to compile.

    Returns:
        List[RdnaNode]: A list of synthesized physical RDNA instructions, comments, or macros.
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
        output_nodes.append(RdnaComment(text=f"Input {var_name} -> {reg}"))

      # --- Outputs ---
      elif node.kind == "Output":
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_vector_register(sources[0])
          output_nodes.append(RdnaComment(text=f"Return: {src_reg}"))

      else:
        # Resolve Abstract ID
        defn = self.semantics.get_definition(node.kind)
        abstract_id = defn[0] if defn else node.kind

        # --- Macro Expansion ---
        if abstract_id in self.macro_registry:
          expander = self.macro_registry[abstract_id]
          kernel_nodes = expander(self.allocator, node.id, node.metadata)
          output_nodes.extend(kernel_nodes)
          continue

        suffix_id = abstract_id.split(".")[-1] if abstract_id else ""
        if suffix_id and suffix_id in self.macro_registry:
          expander = self.macro_registry[suffix_id]
          kernel_nodes = expander(self.allocator, node.id, node.metadata)
          output_nodes.extend(kernel_nodes)
          continue

        # --- 1:1 RdnaInstruction Synthesis ---
        variant = None
        if abstract_id:
          variant = self.semantics.resolve_variant(abstract_id, "rdna")

        if not variant or not variant.get("api"):
          output_nodes.append(RdnaComment(text=f"Unmapped Op: {node.kind} ({node.id})"))
          continue

        opcode = variant["api"]

        # RDNA Vector ALU Format: OPCODE DST, SRC0, SRC1
        dst_reg = self.allocator.get_vector_register(node.id)
        operands: List[RdnaOperand] = [dst_reg]
        sources = input_map.get(node.id, [])

        for src_id in sources:
          # Assume inputs are in VGPRs for ALU ops
          src_reg = self.allocator.get_vector_register(src_id)
          operands.append(src_reg)

        inst = RdnaInstruction(opcode=opcode, operands=operands)
        output_nodes.append(inst)

    return output_nodes

  def to_python(self, rdna_nodes: List[RdnaNode]) -> cst.Module:
    """Converts RDNA AST nodes into a Python source structure representation.

    Args:
        rdna_nodes: A list of RDNA AST nodes to be converted.

    Returns:
        cst.Module: A LibCST Module node representing the generated Python code.
    """
    body_stmts = []

    for node in rdna_nodes:
      stmt = None
      if isinstance(node, RdnaInstruction):
        stmt = self._convert_instruction_to_py(node)
      elif isinstance(node, RdnaLabel):
        stmt = cst.SimpleStatementLine(
          body=[cst.Pass()],
          trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(f"# RdnaLabel: {node.name}")),
        )

      if stmt:
        body_stmts.append(stmt)

    return cst.Module(body=body_stmts)

  def _convert_instruction_to_py(self, inst: RdnaInstruction) -> cst.SimpleStatementLine:
    """Convert RDNA instruction to Python AST.

    Args:
        inst: The RDNA Instruction.

    Returns:
        A CST statement.
    """
    if not inst.operands:
      call = self._make_call(inst.opcode, [])
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    # Heuristic: First operand is destination if it is a register and not a store/branch op
    is_store = "store" in inst.opcode
    is_branch = "branch" in inst.opcode

    dest: Optional[RdnaOperand] = None
    srcs: List[RdnaOperand] = []

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

    if dest and isinstance(dest, (RdnaVGPR, RdnaSGPR)):
      target_name = str(dest)
      # Sanitize brackets for variable names
      clean_target = target_name.replace("[", "_").replace("]", "").replace(":", "_")
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(clean_target))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    else:
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: RdnaOperand) -> cst.BaseExpression:
    """Convert RDNA operand to Python expression.

    Args:
        op: The RDNA operand.

    Returns:
        A CST Expression.
    """
    if isinstance(op, RdnaImmediate):
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
    """Make a Python function call for an instruction.

    Args:
        opcode: The RDNA opcode.
        args: List of CST arguments.

    Returns:
        A CST Call node.
    """
    return cst.Call(func=cst.Attribute(value=cst.Name("rdna"), attr=cst.Name(opcode)), args=args)


class RdnaBackend(CompilerBackend):
  """Compiler Backend implementation for AMD RDNA.

  Orchestrates the synthesis (Graph -> AST) and emission (AST -> Text).
  """

  def __init__(self, semantics: Optional["SemanticsManager"] = None) -> None:
    """Initialize RDNA Backend.

    Args:
        semantics: Optional SemanticsManager.
    """
    # Lazy load if not provided, but typically passed from Registry/Engine
    if semantics is None:
      from ml_switcheroo.semantics.manager import SemanticsManager

      semantics = SemanticsManager()

    self.synthesizer = RdnaSynthesizer(semantics)
    self.emitter = RdnaEmitter()
    # Default architecture for header generation matching legacy adapter defaults
    self.target_arch = "gfx1030"

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles LogicalGraph to RDNA Assembly string.

    Args:
        graph: The intermediate representation.

    Returns:
        str: The RDNA code.
    """
    rdna_nodes = self.synthesizer.from_graph(graph)
    body = self.emitter.emit(rdna_nodes)
    header = f"; RDNA Code Generation Initialized (Arch: {self.target_arch})\n"
    return header + body
