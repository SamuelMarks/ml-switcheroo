"""
SASS Synthesizer and Register Allocator.

This module provides the "Middle-End" logic for the SASS compiler pipeline.
It bridges the gap between high-level Abstract Logic (LogicalGraphs/Python AST)
and low-level Physical Assembly (Instruction nodes/Registers).

It contains:
1.  **RegisterAllocator**: Map Symbolic Variables (e.g., 'x', 'bias') to
    Physical Registers (e.g., 'R0', 'R1').
2.  **SassSynthesizer**:
    -   **Target Transformation (`from_graph`)**: Converts a topological logical graph
        into a linear list of SASS instructions. Supports 1:1 opcode mapping via
        semantics and 1:N expansion via Kernel Macros (e.g. Conv2d loops).
    -   **Source Transformation (`to_python`)**: Converts SASS AST nodes back into
        Python LibCST nodes for high-level analysis or documentation.
"""

from typing import Dict, List, Optional, Union, Callable
import libcst as cst

from ml_switcheroo.core.sass.nodes import (
  Instruction,
  Register,
  Immediate,
  SassNode,
  Comment,
  Operand,
  Label,
)
from ml_switcheroo.core.graph import LogicalGraph, topological_sort
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.sass.macros import expand_conv2d, expand_linear

# Maximum number of general-purpose 32-bit registers per thread in CUDA
MAX_REGISTERS = 255


class RegisterAllocator:
  """
  Manages the mapping between symbolic variable names and physical registers.

  Implements a simple linear allocation strategy (bump pointer). A more complex
  implementation would handle liveness analysis and register spilling, but this
  suffices for basic block translation.
  """

  def __init__(self) -> None:
    """Initializes the allocator with an empty map and counter."""
    self._var_to_reg: Dict[str, str] = {}
    self._next_idx = 0

  def get_register(self, var_name: str) -> Register:
    """
    Retrieves or allocates a register for a symbolic variable.

    If the variable has been seen before, returns the existing register mapping.
    If new, allocates the next available physical register.

    Args:
        var_name (str): The logical identifier (e.g., 'input_1', 'bias').

    Returns:
        Register: A populated Register node (e.g., Register('R0')).

    Raises:
        ValueError: If the allocator runs out of physical registers (>255).
    """
    if var_name in self._var_to_reg:
      return Register(self._var_to_reg[var_name])

    if self._next_idx > MAX_REGISTERS:
      raise ValueError(f"Register overflow! Exceeded {MAX_REGISTERS} registers.")

    reg_name = f"R{self._next_idx}"
    self._var_to_reg[var_name] = reg_name
    self._next_idx += 1
    return Register(reg_name)

  def allocate_temp(self) -> Register:
    """
    Allocates a temporary anonymous register.

    Useful for intermediate calculations or immediate loading.

    Returns:
        Register: A new physical register.
    """
    # Generate a unique temp name
    temp_name = f"__temp_{self._next_idx}__"
    return self.get_register(temp_name)

  def reset(self) -> None:
    """Resets the allocator state."""
    self._var_to_reg.clear()
    self._next_idx = 0


class SassSynthesizer:
  """
  Bidirectional transpiler component.

  Handles:
  1.  **Forward (Graph -> SASS)**: Synthesizes Assembly from Logical Graphs.
      Delegates high-level ops (Conv2d, Linear) to Macros, and low-level ops
      (Add, Mul) to Semantic Opcode Lookup.
  2.  **Reverse (SASS -> Python)**: Synthesizes Python AST from Assembly nodes.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the synthesizer.

    Args:
        semantics (SemanticsManager): The knowledge base for Opcode lookups.
    """
    self.semantics = semantics
    self.allocator = RegisterAllocator()

    # Registry of Kernel Macros for 1-to-N expansion
    # Maps Abstract Operation IDs to expansion functions
    self.macro_registry: Dict[str, Callable] = {
      "Conv2d": expand_conv2d,
      "Linear": expand_linear,
    }

  def from_graph(self, graph: LogicalGraph) -> List[SassNode]:
    """
    Converts a LogicalGraph into a list of SASS AST nodes.

    Process:
    1.  Sorts nodes topologically.
    2.  Traverses nodes.
    3.  For each node:
        a. Check if it matches a Macro (e.g. Conv2d). If so, expand kernel.
        b. If not, lookup abstract opcode mapping (e.g. `Add` -> `FADD`).
        c. Allocate/Resolve Input Registers.
        d. Allocate Output Register.
        e. Construct `Instruction` node.
    4.  Handles `Input` nodes by pre-allocating registers (Contract: R0, R1...).

    Args:
        graph (LogicalGraph): The input computation graph.

    Returns:
        List[SassNode]: A structured list of assembly nodes.
    """
    self.allocator.reset()
    output_nodes: List[SassNode] = []

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
      # Special Handling for Inputs: Just allocate to stabilize register index
      if node.kind == "Input":
        reg = self.allocator.get_register(node.id)
        # Extract original variable name from metadata if available
        var_name = node.metadata.get("name", node.id)
        output_nodes.append(Comment(f"Input {var_name} -> {reg.name}"))
        continue

      if node.kind == "Output":
        # Output nodes are usually sinks, just comment on location
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_register(sources[0])
          output_nodes.append(Comment(f"Return: {src_reg.name}"))
        continue

      # Look up Abstract ID
      # 1. Try treating node.kind as an API path (e.g. "torch.nn.Conv2d")
      # to find Abstract ID ("Conv2d")
      defn = self.semantics.get_definition(node.kind)
      abstract_id = None
      if defn:
        abstract_id = defn[0]
      else:
        # 2. Try treating node.kind as Abstract ID directly
        abstract_id = node.kind

      # --- Macro Expansion Path ---
      if abstract_id in self.macro_registry:
        expander = self.macro_registry[abstract_id]
        # Expand macro using the Allocator protocol.
        # Note: Macros handle their own internal register allocation for loops/etc.
        kernel_nodes = expander(self.allocator, node.id, node.metadata)
        output_nodes.extend(kernel_nodes)
        continue

      # --- 1:1 Instruction Path ---

      # 3. Resolve SASS variant opcode
      variant = None
      if abstract_id:
        variant = self.semantics.resolve_variant(abstract_id, "sass")

      if not variant or not variant.get("api"):
        # Fallback: Emit comment for unmapped op
        output_nodes.append(Comment(f"Unmapped Op: {node.kind} ({node.id})"))
        continue

      opcode = variant["api"]

      # Resolve Operands
      # SASS Convention: OPCODE DST, SRC1, SRC2
      # DST is the register assigned to the current node
      dst_reg = self.allocator.get_register(node.id)

      operands: List[Operand] = [dst_reg]

      # Sources
      sources = input_map.get(node.id, [])
      for src_id in sources:
        src_reg = self.allocator.get_register(src_id)
        operands.append(src_reg)

      inst = Instruction(opcode=opcode, operands=operands)
      output_nodes.append(inst)

    return output_nodes

  def to_python(self, sass_nodes: List[SassNode]) -> cst.Module:
    """
    Converts SASS AST nodes into a Python source structure representation.

    Used for analysis or round-trip verification. Registers are treated as
    variables. Instructions map to function calls `sass.OPCODE(args)`.

    Structure:
        `R0 = sass.FADD(R1, R2)`

    Args:
        sass_nodes (List[SassNode]): List of parsed SASS nodes.

    Returns:
        cst.Module: A LibCST module containing the Python representation.
    """
    body_stmts = []

    for node in sass_nodes:
      stmt = None
      if isinstance(node, Instruction):
        stmt = self._convert_instruction_to_py(node)
      elif isinstance(node, Comment):
        # LibCST comments attach to statements, not standalone easily in body list
        # We emit a "pass" with comment or just ignore for logic graph
        pass
      elif isinstance(node, Label):
        # Labels usually denote blocks. Python doesn't have labels.
        # We emit a comment marker for clarity in decompilation.
        # To attach comment, we need a node.
        comment_node = cst.EmptyLine(comment=cst.Comment(f"# Label: {node.name}"))
        # LibCST Module body expects Statements, not EmptyLines directly unfortunately in all versions,
        # but we can try attaching to a no-op if strictly validating structure.
        pass

      if stmt:
        body_stmts.append(stmt)

    return cst.Module(body=body_stmts)

  def _convert_instruction_to_py(self, inst: Instruction) -> cst.SimpleStatementLine:
    """
    Helper to convert a single instruction to Python CST.

    Assumes SASS semantics: First literal Dest, rest Sources.
    `OP DST, SRC1, SRC2` -> `DST = sass.OP(SRC1, SRC2)`

    Args:
        inst (Instruction): The SASS instruction node.

    Returns:
        cst.SimpleStatementLine: Python statement.
    """
    # SASS usually has DST as op 0.
    if not inst.operands:
      # Side-effect op (e.g. BRA, EXIT, NOP)
      # plain call: sass.OP()
      call = self._make_call(inst.opcode, [])
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    # Determine Dest vs Src
    # Heuristic: If >1 operand, first is Dest.
    dest: Optional[Union[Register, Operand]] = None
    srcs: List[Operand] = []

    # Some ops like ST (Store) don't have dest register, they have side effects on memory.
    # Check semantic knowledge? For now generic heuristic:
    # Standard arithmetic (FADD, FMUL, IMAD) has dest.
    # Control flow (BRA) has no dest.
    # Memory Store (ST) has no register dest.

    is_store = inst.opcode.startswith("ST")
    is_branch = inst.opcode in ["BRA", "BRX", "EXIT", "RET"]
    is_cmp = inst.opcode.startswith("ISETP") or inst.opcode.startswith("ISETP")

    # ISETP typically writes to Predicate register P0
    if is_store or is_branch:
      srcs = inst.operands
    else:
      dest = inst.operands[0]
      srcs = inst.operands[1:]

    # Build Call Args
    arg_nodes = []
    for op in srcs:
      arg_val = self._convert_operand_to_py(op)
      arg_nodes.append(cst.Arg(value=arg_val))

    # Add Predicate as arg if present
    if inst.predicate:
      arg_nodes.append(cst.Arg(keyword=cst.Name("predicate"), value=cst.SimpleString(f"'{inst.predicate}'")))

    call = self._make_call(inst.opcode, arg_nodes)

    # Build Assignment or Expression
    if dest:
      # R0 = ...
      target_name = str(dest)
      # handle register modifiers in assignment target? -R0 = ... is invalid valid.
      # Strip modifiers for LHS
      if isinstance(dest, Register):
        target_name = dest.name

      # SimpleAssignment
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(target_name))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    else:
      # Expression Statement
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: Operand) -> cst.BaseExpression:
    """
    Helper to convert operands to Python Literals/Names.

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

    # Registers, Memory, Predicates -> String Representation -> Name
    # e.g. R0, c[0x0], @P0
    # We sanitize strings to be valid python identifiers if possible,
    # or string literals if complex structure.
    # Registers (R0) are valid IDs. Memory ([R0]) is not.

    raw = str(op)
    if raw.isalnum():
      return cst.Name(raw)

    # Fallback for complex operands (Memory, Negated Regs): return as String Literal
    return cst.SimpleString(f"'{raw}'")

  def _make_call(self, opcode: str, args: List[cst.Arg]) -> cst.Call:
    """
    Constructs `sass.OPCODE(...)`.

    Args:
        opcode (str): The instruction mnemonic.
        args (List[cst.Arg]): Arguments for the call.

    Returns:
        cst.Call: The constructed call node.
    """
    return cst.Call(func=cst.Attribute(value=cst.Name("sass"), attr=cst.Name(opcode)), args=args)
