"""NVIDIA_SASS Synthesizer and NvidiaSassRegister Allocator.

This module provides the "Middle-End" logic for the NVIDIA_SASS compiler pipeline.
It bridges the gap between high-level Abstract Logic (LogicalGraph)
and low-level Physical Assembly (NvidiaSassInstruction nodes/Registers).

It contains:

 **RegisterAllocator**: Map Symbolic Variables (e.g., 'x', 'bias') to
    Physical Registers (e.g., 'R0', 'R1').

 **NvidiaSassSynthesizer**:

**Target Transformation (`from_graph`)**: Converts a topological logical graph
        into a linear list of NVIDIA_SASS instructions. Supports 1:1 opcode mapping via
        semantics and 1:N expansion via Kernel Macros (e.g. Conv2d loops).

**Source Transformation (`to_python`)**: Converts NVIDIA_SASS AST nodes back into
        Python LibCST nodes for high-level analysis or documentation.

 **NvidiaSassBackend**: The CompilerBackend adapter for the Registry.
"""

from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any
import libcst as cst

# Direct Import from Frontend to avoid circular dependency via core shims
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassInstruction,
  NvidiaSassRegister,
  NvidiaSassImmediate,
  NvidiaSassNode,
  NvidiaSassComment,
  NvidiaSassOperand,
  NvidiaSassLabel,
)

# Import IR directly to avoid parsing overhead and cycles with core.graph
from ml_switcheroo.core.compiler.ir import LogicalGraph, topological_sort
import ml_switcheroo.core.compiler.backends.nvidia_sass.macros as sass_macros
import json
import os
import yaml

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

# Maximum number of general-purpose 32-bit registers per thread in CUDA
MAX_REGISTERS = 255


class RegisterAllocator:
  """Manage the mapping between symbolic variable names and physical registers.

  Implements liveness analysis and register spilling, freeing registers
  back to a pool when variables are no longer referenced in the graph.
  """

  def __init__(self) -> None:
    """Initialize the allocator with a free pool."""
    self._var_to_reg: Dict[str, str] = {}
    self._free_pool: List[str] = [f"R{i}" for i in range(MAX_REGISTERS)]
    self._liveness_map: Dict[str, int] = {}

  def free_register(self, var_name: str) -> None:
    """Free a register back to the pool.

    Args:
        var_name (str): The symbolic name of the variable whose register is to be freed.

    """
    if var_name in self._var_to_reg:
      reg = self._var_to_reg.pop(var_name)
      self._free_pool.append(reg)

  def get_register(self, var_name: str) -> NvidiaSassRegister:
    """Retrieve or allocates a register for a symbolic variable.

    Args:
        var_name (str): The symbolic variable name to resolve to a physical register.

    Returns:
        NvidiaSassRegister: The allocated physical register.

    Raises:
        ValueError: If there are no more free physical registers available (overflow).
    """
    if var_name in self._var_to_reg:
      return NvidiaSassRegister(name=self._var_to_reg[var_name])

    if not self._free_pool:
      raise ValueError(f"NvidiaSassRegister overflow! Exceeded {MAX_REGISTERS} registers.")

    reg_name = self._free_pool.pop(0)
    self._var_to_reg[var_name] = reg_name
    return NvidiaSassRegister(name=reg_name)

  def allocate_temp(self) -> NvidiaSassRegister:
    """Allocate a temporary anonymous register.

    Returns:
        NvidiaSassRegister: A unique, newly allocated temporary register.
    """
    import uuid

    temp_name = f"__temp_{uuid.uuid4().hex}__"
    return self.get_register(temp_name)

  def reset(self) -> None:
    """Reset the allocator state.

    Clears all symbolic-to-physical mappings, reinitializes the register pool,
    and clears the liveness tracking map.

    """
    self._var_to_reg.clear()
    self._free_pool = [f"R{i}" for i in range(MAX_REGISTERS)]
    self._liveness_map.clear()

  def build_liveness(self, graph: LogicalGraph) -> None:
    """Build the initial liveness map based on node usage counts.

    Args:
        graph (LogicalGraph): The logical computation graph to analyze.

    """
    self._liveness_map.clear()
    for edge in graph.edges:
      if edge.source not in self._liveness_map:
        self._liveness_map[edge.source] = 0
      self._liveness_map[edge.source] += 1

  def record_usage(self, var_name: str) -> None:
    """Record a usage and frees the register if it's the last one.

    Args:
        var_name (str): The name of the variable being referenced.

    """
    if var_name in self._liveness_map:
      self._liveness_map[var_name] -= 1
      if self._liveness_map[var_name] <= 0:
        self.free_register(var_name)


class NvidiaSassSynthesizer:
  """Bidirectional transpiler component.

   Handles:

  **Forward (Graph -> NVIDIA_SASS)**: Synthesizes Assembly from Logical Graphs.
       Delegates high-level ops (Conv2d, Linear) to Macros, and low-level ops
       (Add, Mul) to Semantic Opcode Lookup.

  **Reverse (NVIDIA_SASS -> Python)**: Synthesizes Python AST from Assembly nodes.
  """

  def __init__(self, semantics: "SemanticsManager"):
    """Initialize the synthesizer.

    Args:
        semantics (SemanticsManager): The knowledge base for Opcode lookups.

    """
    self.semantics = semantics
    self.allocator = RegisterAllocator()

    # Registry of Kernel Macros for 1-to-N expansion
    # Maps Abstract Operation IDs to expansion functions
    self.macro_registry = {}
    macros_yaml_path = os.path.join(os.path.dirname(__file__), "macros.yaml")
    macros_json_path = os.path.join(os.path.dirname(__file__), "macros.json")
    mapping: Dict[str, Any] = {}
    if os.path.exists(macros_yaml_path):
      try:
        with open(macros_yaml_path, "r", encoding="utf-8") as f:
          mapping = yaml.safe_load(f) or {}
      except FileNotFoundError:
        mapping = {}
    if not mapping and os.path.exists(macros_json_path):
      try:
        with open(macros_json_path, "r", encoding="utf-8") as f:
          mapping = json.load(f)
      except FileNotFoundError:
        mapping = {}

    for key, func_name in mapping.items():
      if hasattr(sass_macros, func_name):
        self.macro_registry[key] = getattr(sass_macros, func_name)

  def from_graph(self, graph: LogicalGraph) -> List[NvidiaSassNode]:
    """Convert a LogicalGraph into a list of NVIDIA_SASS AST nodes.

       Process:

    Sorts nodes topologically.

    Traverses nodes.

    For each node:
           a. Check if it matches a Macro (e.g. Conv2d). If so, expand kernel.
           b. If not, lookup abstract opcode mapping (e.g. `Add` -> `FADD`).
           c. Allocate/Resolve Input Registers.
           d. Allocate Output NvidiaSassRegister.
           e. Construct `NvidiaSassInstruction` node.

    Handles `Input` nodes by pre-allocating registers (Contract: R0, R1...).

    Args:
           graph (LogicalGraph): The input computation graph.

    Returns:
           List[~ml_switcheroo.core.compiler.frontends.nvidia_sass.cst.NvidiaSassNode]: A structured list of assembly nodes.

    """
    self.allocator.reset()
    self.allocator.build_liveness(graph)
    output_nodes: List[NvidiaSassNode] = []

    # 1. Topological Sort ensures dependencies are met
    sorted_nodes = topological_sort(graph)

    # 2. Build adjacency map for inputs (Edges point Node -> Node)
    # We need to look up which nodes feed INTO current node
    # input_map: {target_id: [source_id_0, source_id_1]}
    input_map: Dict[str, List[str]] = {}
    for edge in graph.edges:
      if edge.target not in input_map:
        input_map[edge.target] = []
      input_map[edge.target].append(edge.source)

    for node in sorted_nodes:
      op_type: str = str(node.op_type if hasattr(node, "op_type") else getattr(node, "kind", ""))
      node_attrs = getattr(node, "attributes", {})

      # Special Handling for Inputs: Just allocate to stabilize register index
      if op_type == "Input":
        reg = self.allocator.get_register(node.id)
        # Extract original variable name from attributes if available
        var_name = node_attrs.get("name", node.id)
        output_nodes.append(NvidiaSassComment(text=f"Input {var_name} -> {reg.name}"))

      elif op_type == "Output":
        # Output nodes are usually sinks, just comment on location
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_register(sources[0])
          output_nodes.append(NvidiaSassComment(text=f"Return: {src_reg.name}"))
          self.allocator.record_usage(sources[0])

      else:
        # Look up Abstract ID
        # 1. Try treating op_type as an API path (e.g. "torch.nn.Conv2d")
        # to find Abstract ID ("Conv2d")
        defn = self.semantics.get_definition(op_type)
        if not defn and ("." in op_type):
          suffix = op_type.split(".", 1)[-1]
          defn = self.semantics.get_definition(suffix)
        abstract_id = None
        if defn:
          abstract_id = defn[0]
        else:
          # 2. Try treating op_type as Abstract ID directly
          abstract_id = op_type

        # --- Macro Expansion Path ---
        if abstract_id in self.macro_registry:
          expander = self.macro_registry[abstract_id]
          # Expand macro using the Allocator protocol.
          # Note: Macros handle their own internal register allocation for loops/etc.
          kernel_nodes = expander(self.allocator, node.id, node_attrs)
          output_nodes.extend(kernel_nodes)
          sources = input_map.get(node.id, [])
          for src_id in sources:
            self.allocator.record_usage(src_id)
          continue

        # Try suffix macro match for method calls like `hidden_states.reshape`
        suffix_id = abstract_id.split(".")[-1] if abstract_id else ""
        if suffix_id and suffix_id in self.macro_registry:
          expander = self.macro_registry[suffix_id]
          kernel_nodes = expander(self.allocator, node.id, node_attrs)
          output_nodes.extend(kernel_nodes)
          sources = input_map.get(node.id, [])
          for src_id in sources:
            self.allocator.record_usage(src_id)
          continue

        # --- 1:1 NvidiaSassInstruction Path ---

        # 3. Resolve NVIDIA_SASS variant opcode
        variant = None
        if abstract_id:
          variant = self.semantics.resolve_variant(abstract_id, "nvidia_sass")

        if not variant or not variant.get("api"):
          # Fallback: Emit comment for unmapped op
          node_op_type = node.op_type if hasattr(node, "op_type") else getattr(node, "kind", "")
          output_nodes.append(NvidiaSassComment(text=f"Unmapped Op: {node_op_type} ({node.id})"))
          continue

        opcode = variant["api"]

        # Resolve Operands
        # NVIDIA_SASS Convention: OPCODE DST, SRC1, SRC2
        # DST is the register assigned to the current node
        dst_reg = self.allocator.get_register(node.id)

        operands: List[NvidiaSassOperand] = [dst_reg]

        # Sources
        sources = input_map.get(node.id, [])
        for src_id in sources:
          src_reg = self.allocator.get_register(src_id)
          operands.append(src_reg)
          self.allocator.record_usage(src_id)

        inst = NvidiaSassInstruction(opcode=opcode, operands=operands)
        output_nodes.append(inst)

    return output_nodes

  def to_python(self, sass_nodes: List[NvidiaSassNode]) -> cst.Module:
    """Convert NVIDIA_SASS AST nodes into a Python source structure representation.

    Used for analysis or round-trip verification. Registers are treated as
    variables. Instructions map to function calls `nvidia_sass.OPCODE(args)`.

    Structure:
        `R0 = nvidia_sass.FADD(R1, R2)`

    Args:
        sass_nodes (List[~ml_switcheroo.core.compiler.frontends.nvidia_sass.cst.NvidiaSassNode]): List of parsed NVIDIA_SASS nodes.

    Returns:
        cst.Module: A LibCST module containing the Python representation.

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
        # Labels usually denote blocks. Python doesn't have labels.
        # We emit a comment marker for clarity in decompilation.
        # To attach comment, we need a node.
        stmt = cst.SimpleStatementLine(
          body=[cst.Pass()],
          trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(value=f"# NvidiaSassLabel: {node.name}")),
        )

      if stmt:
        body_stmts.append(stmt)

    return cst.Module(body=body_stmts)

  def _convert_instruction_to_py(self, inst: NvidiaSassInstruction) -> cst.SimpleStatementLine:
    """Support to convert a single instruction to Python CST.

    Assumes NVIDIA_SASS semantics: First literal Dest, rest Sources.
    `OP DST, SRC1, SRC2` -> `DST = nvidia_sass.OP(SRC1, SRC2)`

    Args:
        inst (NvidiaSassInstruction): The NVIDIA_SASS instruction node.

    Returns:
        cst.SimpleStatementLine: Python statement.

    """
    # NVIDIA_SASS usually has DST as op 0.
    if not inst.operands:
      # Side-effect op (e.g. BRA, EXIT, NOP)
      # plain call: nvidia_sass.OP()
      call = self._make_call(inst.opcode, [])
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    # Determine Dest vs Src
    # Heuristic: If >1 operand, first is Dest.
    dest: Optional[Union[NvidiaSassRegister, NvidiaSassOperand]] = None
    srcs: List[NvidiaSassOperand] = []

    # Some ops like ST (Store) don't have dest register, they have side effects on memory.
    # Check semantic knowledge? For now generic heuristic:
    # Standard arithmetic (FADD, FMUL, IMAD) has dest.
    # Control flow (BRA) has no dest.
    # NvidiaSassMemory Store (ST) has no register dest.

    is_store = inst.opcode.startswith("ST")
    is_branch = inst.opcode in ["BRA", "BRX", "EXIT", "RET"]
    is_nop = inst.opcode == "NOP"
    # is_cmp = inst.opcode.startswith("ISETP") or inst.opcode.startswith("ISETP")

    # ISETP typically writes to NvidiaSassPredicate register P0
    if is_store or is_branch or is_nop:
      srcs = inst.operands
    else:
      dest = inst.operands[0]
      srcs = inst.operands[1:]

    # Build Call Args
    arg_nodes = []
    for op in srcs:
      arg_val = self._convert_operand_to_py(op)
      arg_nodes.append(cst.Arg(value=arg_val))

    # Add NvidiaSassPredicate as arg if present
    if inst.predicate:
      arg_nodes.append(cst.Arg(keyword=cst.Name("predicate"), value=cst.SimpleString(f"'{inst.predicate}'")))

    call = self._make_call(inst.opcode, arg_nodes)

    # Build Assignment or Expression
    if dest:
      # R0 = ...
      target_name = str(dest)
      # handle register modifiers in assignment target? -R0 = ... is invalid valid.
      if not target_name.isidentifier():
        return cst.SimpleStatementLine(body=[cst.Expr(value=call)])
      # Strip modifiers for LHS
      if isinstance(dest, NvidiaSassRegister):
        target_name = dest.name

      # SimpleAssignment
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(target_name))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    else:
      # Expression Statement
      pass
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: NvidiaSassOperand) -> cst.BaseExpression:
    """Support to convert operands to Python Literals/Names.

    Args:
        op (NvidiaSassOperand): The operand node.

    Returns:
        cst.BaseExpression: The corresponding Python AST node.

    """
    if isinstance(op, NvidiaSassImmediate):
      if op.is_hex:
        return cst.Integer(hex(int(op.value)))
      if isinstance(op.value, float):
        return cst.Float(str(op.value))
      return cst.Integer(str(int(op.value)))

    # Registers, NvidiaSassMemory, Predicates -> String Representation -> Name
    # e.g. R0, c[0x0], @P0
    # We sanitize strings to be valid python identifiers if possible,
    # or string literals if complex structure.
    # Registers (R0) are valid IDs. NvidiaSassMemory ([R0]) is not.

    raw = str(op)
    if raw.isalnum() and not raw.isdigit():
      return cst.Name(raw)

    # Fallback for complex operands (NvidiaSassMemory, Negated Regs): return as String Literal
    return cst.SimpleString(f"'{raw}'")

  def _make_call(self, opcode: str, args: List[cst.Arg]) -> cst.Call:
    """Construct a `nvidia_sass.OPCODE(...)` function call in Python CST.

    Args:
        opcode (str): The name of the NVIDIA_SASS operation.
        args (List[cst.Arg]): The arguments to pass to the function call.

    Returns:
        cst.Call: The constructed LibCST Call expression.
    """
    return cst.Call(func=cst.Attribute(value=cst.Name("nvidia_sass"), attr=cst.Name(opcode)), args=args)
