"""
RDNA Synthesizer (Backend).

Converts LogicalGraph IR into RDNA AST nodes.
Implements compiler backend interface.
"""

from typing import Callable, Dict, List, Any
from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.ir import LogicalGraph, topological_sort
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.compiler.frontends.rdna.nodes import (
  Comment,
  Instruction,
  Operand,
  RdnaNode,
  SGPR,
  VGPR,
  Immediate,
  Label,
)
from ml_switcheroo.compiler.backends.rdna.macros import expand_conv2d, expand_linear
from ml_switcheroo.compiler.backends.rdna.emitter import RdnaEmitter
import libcst as cst

MAX_VGPR = 256
MAX_SGPR = 106


class RegisterAllocator:
  """Manages mapping between symbolic names and physical registers."""

  def __init__(self) -> None:
    self._var_to_vgpr: Dict[str, int] = {}
    self._var_to_sgpr: Dict[str, int] = {}
    self._next_vgpr = 0
    self._next_sgpr = 0

  def get_vector_register(self, var_name: str) -> VGPR:
    """Gets or allocates a VGPR."""
    if var_name in self._var_to_vgpr:
      return VGPR(self._var_to_vgpr[var_name])

    if self._next_vgpr >= MAX_VGPR:
      raise ValueError(f"VGPR overflow! Exceeded {MAX_VGPR} registers.")

    idx = self._next_vgpr
    self._var_to_vgpr[var_name] = idx
    self._next_vgpr += 1
    return VGPR(idx)

  def get_scalar_register(self, var_name: str) -> SGPR:
    """Gets or allocates an SGPR."""
    if var_name in self._var_to_sgpr:
      return SGPR(self._var_to_sgpr[var_name])

    if self._next_sgpr >= MAX_SGPR:
      raise ValueError(f"SGPR overflow! Exceeded {MAX_SGPR} registers.")

    idx = self._next_sgpr
    self._var_to_sgpr[var_name] = idx
    self._next_sgpr += 1
    return SGPR(idx)

  def allocate_vector_temp(self) -> VGPR:
    """Allocates temp VGPR."""
    name = f"__v_temp_{self._next_vgpr}__"
    return self.get_vector_register(name)

  def allocate_scalar_temp(self) -> SGPR:
    """Allocates temp SGPR."""
    name = f"__s_temp_{self._next_sgpr}__"
    return self.get_scalar_register(name)

  def reset(self) -> None:
    self._var_to_vgpr.clear()
    self._var_to_sgpr.clear()
    self._next_vgpr = 0
    self._next_sgpr = 0


class RdnaSynthesizer:
  """Core logic for Graph -> RDNA AST transformation."""

  def __init__(self, semantics: SemanticsManager) -> None:
    self.semantics = semantics
    self.allocator = RegisterAllocator()
    self.macro_registry: Dict[str, Callable] = {
      "Conv2d": expand_conv2d,
      "Linear": expand_linear,
    }

  def from_graph(self, graph: LogicalGraph) -> List[RdnaNode]:
    """Converts LogicalGraph to RDNA AST nodes."""
    return self.synthesize(graph)

  def synthesize(self, graph: LogicalGraph) -> List[RdnaNode]:
    """Converts LogicalGraph to RDNA AST nodes."""
    self.allocator.reset()
    output_nodes: List[RdnaNode] = []
    sorted_nodes = topological_sort(graph)

    input_map: Dict[str, List[str]] = {}
    for edge in graph.edges:
      if edge.target not in input_map:
        input_map[edge.target] = []
      input_map[edge.target].append(edge.source)

    for node in sorted_nodes:
      if node.kind == "Input":
        reg = self.allocator.get_vector_register(node.id)
        var_name = node.metadata.get("name", node.id)
        output_nodes.append(Comment(f"Input {var_name} -> {reg}"))
        continue

      if node.kind == "Output":
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_vector_register(sources[0])
          output_nodes.append(Comment(f"Return: {src_reg}"))
        continue

      defn = self.semantics.get_definition(node.kind)
      abstract_id = defn[0] if defn else node.kind

      if abstract_id in self.macro_registry:
        expander = self.macro_registry[abstract_id]
        kernel_nodes = expander(self.allocator, node.id, node.metadata)
        output_nodes.extend(kernel_nodes)
        continue

      variant = None
      if abstract_id:
        variant = self.semantics.resolve_variant(abstract_id, "rdna")

      if not variant or not variant.get("api"):
        output_nodes.append(Comment(f"Unmapped Op: {node.kind} ({node.id})"))
        continue

      opcode = variant["api"]
      dst_reg = self.allocator.get_vector_register(node.id)
      operands: List[Operand] = [dst_reg]
      sources = input_map.get(node.id, [])

      for src_id in sources:
        operands.append(self.allocator.get_vector_register(src_id))

      inst = Instruction(opcode=opcode, operands=operands)
      output_nodes.append(inst)

    return output_nodes

  def to_python(self, rdna_nodes: List[RdnaNode]) -> cst.Module:
    """
    Converts RDNA AST nodes into a Python source structure representation.
    Used for analysis or round-trip verification.

    Structure: `v0 = rdna.v_add_f32(v1, v2)`

    Args:
        rdna_nodes (List[RdnaNode]): List of parsed RDNA nodes.

    Returns:
        cst.Module: A LibCST module containing the Python representation.
    """
    body_stmts = []

    for node in rdna_nodes:
      stmt = None
      if isinstance(node, Instruction):
        stmt = self._convert_instruction_to_py(node)
      elif isinstance(node, Label):
        # Labels are blocks markers
        stmt = cst.SimpleStatementLine(
          body=[cst.Pass()],
          trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(f"# Label: {node.name}")),
        )

      if stmt:
        body_stmts.append(stmt)

    return cst.Module(body=body_stmts)

  def _convert_instruction_to_py(self, inst: Instruction) -> cst.SimpleStatementLine:
    """
    Helper to convert a single instruction to Python CST.
    Assumes standard RDNA semantics: First operand is Dest.

    Args:
        inst (Instruction): RDNA instruction node.

    Returns:
        cst.SimpleStatementLine: Python statement.
    """
    if not inst.operands:
      call = self._make_call(inst.opcode, [])
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    # Heuristic: First operand is destination if it is a register and not a store/branch op
    is_store = "store" in inst.opcode
    is_branch = "branch" in inst.opcode
    is_cmp = "cmp" in inst.opcode  # Compares write to VCC/SCC implicitly or explicitly

    dest: Optional[Operand] = None
    srcs: List[Operand] = []

    if is_store or is_branch:
      srcs = inst.operands
    else:
      dest = inst.operands[0]
      srcs = inst.operands[1:]

    # Build Arguments
    arg_nodes = []
    for op in srcs:
      val_node = self._convert_operand_to_py(op)
      arg_nodes.append(cst.Arg(value=val_node))

    call = self._make_call(inst.opcode, arg_nodes)

    if dest and isinstance(dest, (VGPR, SGPR)):
      target_name = str(dest)
      # v[0:3] -> v_0_3 for valid python target?
      # str(dest) gives "v0" or "v[0:3]".
      # We sanitize brackets for variable names.
      clean_target = target_name.replace("[", "_").replace("]", "").replace(":", "_")
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(clean_target))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    else:
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: Operand) -> cst.BaseExpression:
    """
    Helper to convert operand to Python literal/name.

    Args:
        op (Operand): The operand node.

    Returns:
        cst.BaseExpression: The corresponding Python AST node.
    """
    if isinstance(op, Immediate):
      if op.is_hex:
        return cst.Integer(hex(int(op.value)))
      if isinstance(op.value, float):
        return cst.Float(str(op.value))
      return cst.Integer(str(int(op.value)))

    raw = str(op)
    # Sanitize registers range s[0:3] -> s_0_3
    if "[" in raw:
      clean = raw.replace("[", "_").replace("]", "").replace(":", "_")
      return cst.Name(clean)

    if raw.isalnum() or "_" in raw:
      return cst.Name(raw)

    return cst.SimpleString(f"'{raw}'")

  def _make_call(self, opcode: str, args: List[cst.Arg]) -> cst.Call:
    """Constructs `rdna.OPCODE(...)`."""
    return cst.Call(func=cst.Attribute(value=cst.Name("rdna"), attr=cst.Name(opcode)), args=args)


class RdnaBackend(CompilerBackend):
  """
  RDNA Compiler Backend wrapper.
  """

  def __init__(self, semantics: SemanticsManager) -> None:
    self.synthesizer = RdnaSynthesizer(semantics)
    self.emitter = RdnaEmitter()

  def compile(self, graph: LogicalGraph) -> str:
    """
    Compiles Graph to RDNA text.
    """
    nodes = self.synthesizer.synthesize(graph)
    header = "; RDNA Code Generation Initialized (Arch: gfx1030)\n"
    return header + self.emitter.emit(nodes)
