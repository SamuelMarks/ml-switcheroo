"""SASS Synthesizer and SassRegister Allocator.

This module provides the "Middle-End" logic for the SASS compiler pipeline.
It bridges the gap between high-level Abstract Logic (LogicalGraph)
and low-level Physical Assembly (SassInstruction nodes/Registers).

It contains:

 **RegisterAllocator**: Map Symbolic Variables (e.g., 'x', 'bias') to
    Physical Registers (e.g., 'R0', 'R1').

 **SassSynthesizer**:

**Target Transformation (`from_graph`)**: Converts a topological logical graph
        into a linear list of SASS instructions. Supports 1:1 opcode mapping via
        semantics and 1:N expansion via Kernel Macros (e.g. Conv2d loops).

**Source Transformation (`to_python`)**: Converts SASS AST nodes back into
        Python LibCST nodes for high-level analysis or documentation.

 **SassBackend**: The CompilerBackend adapter for the Registry.
"""

from typing import Any

from typing import Dict, List, Optional, Union, Callable, TYPE_CHECKING
import libcst as cst

# Direct Import from Frontend to avoid circular dependency via core shims
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassInstruction,
  SassRegister,
  SassImmediate,
  SassNode,
  SassComment,
  SassOperand,
  SassLabel,
)

# Import IR directly to avoid parsing overhead and cycles with core.graph
from ml_switcheroo.core.compiler.ir import LogicalGraph, topological_sort
from ml_switcheroo.core.compiler.backends.sass.macros import (
  expand_conv2d,
  expand_linear,
  expand_mean,
  expand_relu,
  expand_flatten,
  expand_reshape,
  expand_conv3d,
  expand_avgpool2d,
  expand_maxpool2d,
  expand_batchnorm2d,
  expand_dropout,
  expand_sigmoid,
  expand_tanh,
  expand_gelu,
  expand_mseloss,
  expand_crossentropyloss,
  expand_rnn,
  expand_lstm,
  expand_gru,
  expand_multiheadattention,
  expand_transformer,
  expand_transformerencoder,
  expand_transformerdecoder,
  expand_conv1d,
  expand_depthwiseconv2d,
  expand_convtranspose,
  expand_pool1d,
  expand_pool3d,
  expand_adaptivepool,
  expand_generic_norm,
  expand_generic_activation,
  expand_generic_linalg,
  expand_generic_reduction,
  expand_generic_loss,
  expand_generic_dropout,
)

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

# Maximum number of general-purpose 32-bit registers per thread in CUDA
MAX_REGISTERS = 255


class RegisterAllocator:
  """Manages the mapping between symbolic variable names and physical registers.

  Implements liveness analysis and register spilling, freeing registers
  back to a pool when variables are no longer referenced in the graph.
  """

  def __init__(self) -> None:
    """Initializes the allocator with a free pool."""
    self._var_to_reg: Dict[str, str] = {}
    self._free_pool: List[str] = [f"R{i}" for i in range(MAX_REGISTERS)]
    self._liveness_map: Dict[str, int] = {}

  def free_register(self, var_name: str) -> None:
    """Frees a register back to the pool.

    Args:
        var_name (str): The symbolic name of the variable whose register is to be freed.

    """
    if var_name in self._var_to_reg:
      reg = self._var_to_reg.pop(var_name)
      self._free_pool.append(reg)

  def get_register(self, var_name: str) -> SassRegister:
    """Retrieves or allocates a register for a symbolic variable.

    Args:
        var_name (str): The symbolic variable name to resolve to a physical register.

    Returns:
        SassRegister: The allocated physical register.

    Raises:
        ValueError: If there are no more free physical registers available (overflow).
    """
    if var_name in self._var_to_reg:
      return SassRegister(name=self._var_to_reg[var_name])

    if not self._free_pool:
      raise ValueError(f"SassRegister overflow! Exceeded {MAX_REGISTERS} registers.")

    reg_name = self._free_pool.pop(0)
    self._var_to_reg[var_name] = reg_name
    return SassRegister(name=reg_name)

  def allocate_temp(self) -> SassRegister:
    """Allocates a temporary anonymous register.

    Returns:
        SassRegister: A unique, newly allocated temporary register.
    """
    import uuid

    temp_name = f"__temp_{uuid.uuid4().hex}__"
    return self.get_register(temp_name)

  def reset(self) -> None:
    """Resets the allocator state.

    Clears all symbolic-to-physical mappings, reinitializes the register pool,
    and clears the liveness tracking map.

    """
    self._var_to_reg.clear()
    self._free_pool = [f"R{i}" for i in range(MAX_REGISTERS)]
    self._liveness_map.clear()

  def build_liveness(self, graph: LogicalGraph) -> None:
    """Builds the initial liveness map based on node usage counts.

    Args:
        graph (LogicalGraph): The logical computation graph to analyze.

    """
    self._liveness_map.clear()
    for edge in graph.edges:
      if edge.source not in self._liveness_map:
        self._liveness_map[edge.source] = 0
      self._liveness_map[edge.source] += 1

  def record_usage(self, var_name: str) -> None:
    """Records a usage and frees the register if it's the last one.

    Args:
        var_name (str): The name of the variable being referenced.

    """
    if var_name in self._liveness_map:
      self._liveness_map[var_name] -= 1
      if self._liveness_map[var_name] <= 0:
        self.free_register(var_name)


class SassSynthesizer:
  """Bidirectional transpiler component.

   Handles:

  **Forward (Graph -> SASS)**: Synthesizes Assembly from Logical Graphs.
       Delegates high-level ops (Conv2d, Linear) to Macros, and low-level ops
       (Add, Mul) to Semantic Opcode Lookup.

  **Reverse (SASS -> Python)**: Synthesizes Python AST from Assembly nodes.
  """

  def __init__(self, semantics: "SemanticsManager"):
    """Initializes the synthesizer.

    Args:
        semantics (SemanticsManager): The knowledge base for Opcode lookups.

    """
    self.semantics = semantics
    self.allocator = RegisterAllocator()

    # Registry of Kernel Macros for 1-to-N expansion
    # Maps Abstract Operation IDs to expansion functions
    self.macro_registry: Dict[str, Callable[..., Any]] = {
      "Conv2d": expand_conv2d,
      "Linear": expand_linear,
      "Mean": expand_mean,
      "mean": expand_mean,
      "ReLU": expand_relu,
      "relu": expand_relu,
      "Flatten": expand_flatten,
      "flatten": expand_flatten,
      "Reshape": expand_reshape,
      "reshape": expand_reshape,
      "Conv3d": expand_conv3d,
      "conv3d": expand_conv3d,
      "AvgPool2d": expand_avgpool2d,
      "avgpool2d": expand_avgpool2d,
      "MaxPool2d": expand_maxpool2d,
      "maxpool2d": expand_maxpool2d,
      "BatchNorm2d": expand_batchnorm2d,
      "batchnorm2d": expand_batchnorm2d,
      "Sigmoid": expand_sigmoid,
      "sigmoid": expand_sigmoid,
      "Tanh": expand_tanh,
      "tanh": expand_tanh,
      "MSELoss": expand_mseloss,
      "mseloss": expand_mseloss,
      "RNN": expand_rnn,
      "rnn": expand_rnn,
      "LSTM": expand_lstm,
      "lstm": expand_lstm,
      "MultiheadAttention": expand_multiheadattention,
      "multiheadattention": expand_multiheadattention,
      "Transformer": expand_transformer,
      "transformer": expand_transformer,
      "TransformerEncoder": expand_transformerencoder,
      "transformerencoder": expand_transformerencoder,
      "Conv1d": expand_conv1d,
      "DepthwiseConv2d": expand_depthwiseconv2d,
      "ConvTranspose1d": expand_convtranspose,
      "ConvTranspose2d": expand_convtranspose,
      "ConvTranspose3d": expand_convtranspose,
      "AvgPool1d": expand_pool1d,
      "MaxPool1d": expand_pool1d,
      "AvgPool3d": expand_pool3d,
      "MaxPool3d": expand_pool3d,
      "AdaptiveAvgPool2d": expand_adaptivepool,
      "AdaptiveMaxPool2d": expand_adaptivepool,
      "BatchNorm1d": expand_generic_norm,
      "BatchNorm3d": expand_generic_norm,
      "LayerNorm": expand_generic_norm,
      "GroupNorm": expand_generic_norm,
      "InstanceNorm2d": expand_generic_norm,
      "Softmax": expand_generic_activation,
      "LogSoftmax": expand_generic_activation,
      "SiLU": expand_generic_activation,
      "Swish": expand_generic_activation,
      "ELU": expand_generic_activation,
      "LeakyReLU": expand_generic_activation,
      "BMM": expand_generic_linalg,
      "Dot": expand_generic_linalg,
      "SVD": expand_generic_linalg,
      "Solve": expand_generic_linalg,
      "Cholesky": expand_generic_linalg,
      "Sum": expand_generic_reduction,
      "Prod": expand_generic_reduction,
      "Min": expand_generic_reduction,
      "Max": expand_generic_reduction,
      "ArgMax": expand_generic_reduction,
      "ArgMin": expand_generic_reduction,
      "Any": expand_generic_reduction,
      "All": expand_generic_reduction,
      "BCEWithLogitsLoss": expand_generic_loss,
      "L1Loss": expand_generic_loss,
      "NLLLoss": expand_generic_loss,
      "Dropout2d": expand_generic_dropout,
      "Dropout3d": expand_generic_dropout,
      "AlphaDropout": expand_generic_dropout,
      "TransformerDecoder": expand_transformerdecoder,
      "transformerdecoder": expand_transformerdecoder,
      "GRU": expand_gru,
      "gru": expand_gru,
      "LSTMCell": expand_lstm,
      "GRUCell": expand_gru,
      "CrossEntropyLoss": expand_crossentropyloss,
      "crossentropyloss": expand_crossentropyloss,
      "GELU": expand_gelu,
      "gelu": expand_gelu,
      "Dropout": expand_dropout,
      "dropout": expand_dropout,
      "MatMul": expand_linear,
      "matmul": expand_linear,
    }

  def from_graph(self, graph: LogicalGraph) -> List[SassNode]:
    """Converts a LogicalGraph into a list of SASS AST nodes.

       Process:

    Sorts nodes topologically.

    Traverses nodes.

    For each node:
           a. Check if it matches a Macro (e.g. Conv2d). If so, expand kernel.
           b. If not, lookup abstract opcode mapping (e.g. `Add` -> `FADD`).
           c. Allocate/Resolve Input Registers.
           d. Allocate Output SassRegister.
           e. Construct `SassInstruction` node.

    Handles `Input` nodes by pre-allocating registers (Contract: R0, R1...).

    Args:
           graph (LogicalGraph): The input computation graph.

    Returns:
           List[SassNode]: A structured list of assembly nodes.

    """
    self.allocator.reset()
    self.allocator.build_liveness(graph)
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
        output_nodes.append(SassComment(text=f"Input {var_name} -> {reg.name}"))
        continue

      if node.kind == "Output":
        # Output nodes are usually sinks, just comment on location
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_register(sources[0])
          output_nodes.append(SassComment(text=f"Return: {src_reg.name}"))
          self.allocator.record_usage(sources[0])
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
        sources = input_map.get(node.id, [])
        for src_id in sources:
          self.allocator.record_usage(src_id)
        continue

      # --- 1:1 SassInstruction Path ---

      # 3. Resolve SASS variant opcode
      variant = None
      if abstract_id:
        variant = self.semantics.resolve_variant(abstract_id, "sass")

      if not variant or not variant.get("api"):
        # Fallback: Emit comment for unmapped op
        output_nodes.append(SassComment(text=f"Unmapped Op: {node.kind} ({node.id})"))
        continue

      opcode = variant["api"]

      # Resolve Operands
      # SASS Convention: OPCODE DST, SRC1, SRC2
      # DST is the register assigned to the current node
      dst_reg = self.allocator.get_register(node.id)

      operands: List[SassOperand] = [dst_reg]

      # Sources
      sources = input_map.get(node.id, [])
      for src_id in sources:
        src_reg = self.allocator.get_register(src_id)
        operands.append(src_reg)
        self.allocator.record_usage(src_id)

      inst = SassInstruction(opcode=opcode, operands=operands)
      output_nodes.append(inst)

    return output_nodes

  def to_python(self, sass_nodes: List[SassNode]) -> cst.Module:
    """Converts SASS AST nodes into a Python source structure representation.

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
      if isinstance(node, SassInstruction):
        stmt = self._convert_instruction_to_py(node)
      elif isinstance(node, SassComment):
        if "BEGIN" in node.text or "END" in node.text:
          stmt = cst.SimpleStatementLine(
            body=[cst.Pass()], trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(value=f"# {node.text}"))
          )
      elif isinstance(node, SassLabel):
        # Labels usually denote blocks. Python doesn't have labels.
        # We emit a comment marker for clarity in decompilation.
        # To attach comment, we need a node.
        stmt = cst.SimpleStatementLine(
          body=[cst.Pass()],
          trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(value=f"# SassLabel: {node.name}")),
        )

      if stmt:
        body_stmts.append(stmt)

    return cst.Module(body=body_stmts)

  def _convert_instruction_to_py(self, inst: SassInstruction) -> cst.SimpleStatementLine:
    """Helper to convert a single instruction to Python CST.

    Assumes SASS semantics: First literal Dest, rest Sources.
    `OP DST, SRC1, SRC2` -> `DST = sass.OP(SRC1, SRC2)`

    Args:
        inst (SassInstruction): The SASS instruction node.

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
    dest: Optional[Union[SassRegister, SassOperand]] = None
    srcs: List[SassOperand] = []

    # Some ops like ST (Store) don't have dest register, they have side effects on memory.
    # Check semantic knowledge? For now generic heuristic:
    # Standard arithmetic (FADD, FMUL, IMAD) has dest.
    # Control flow (BRA) has no dest.
    # SassMemory Store (ST) has no register dest.

    is_store = inst.opcode.startswith("ST")
    is_branch = inst.opcode in ["BRA", "BRX", "EXIT", "RET"]
    # is_cmp = inst.opcode.startswith("ISETP") or inst.opcode.startswith("ISETP")

    # ISETP typically writes to SassPredicate register P0
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

    # Add SassPredicate as arg if present
    if inst.predicate:
      arg_nodes.append(cst.Arg(keyword=cst.Name("predicate"), value=cst.SimpleString(f"'{inst.predicate}'")))

    call = self._make_call(inst.opcode, arg_nodes)

    # Build Assignment or Expression
    if dest:
      # R0 = ...
      target_name = str(dest)
      # handle register modifiers in assignment target? -R0 = ... is invalid valid.
      # Strip modifiers for LHS
      if isinstance(dest, SassRegister):
        target_name = dest.name

      # SimpleAssignment
      assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name(target_name))], value=call)
      return cst.SimpleStatementLine(body=[assign])
    else:
      # Expression Statement
      return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

  def _convert_operand_to_py(self, op: SassOperand) -> cst.BaseExpression:
    """Helper to convert operands to Python Literals/Names.

    Args:
        op (SassOperand): The operand node.

    Returns:
        cst.BaseExpression: The corresponding Python AST node.

    """
    if isinstance(op, SassImmediate):
      if op.is_hex:
        return cst.Integer(hex(int(op.value)))
      if isinstance(op.value, float):
        return cst.Float(str(op.value))
      return cst.Integer(str(int(op.value)))

    # Registers, SassMemory, Predicates -> String Representation -> Name
    # e.g. R0, c[0x0], @P0
    # We sanitize strings to be valid python identifiers if possible,
    # or string literals if complex structure.
    # Registers (R0) are valid IDs. SassMemory ([R0]) is not.

    raw = str(op)
    if raw.isalnum():
      return cst.Name(raw)

    # Fallback for complex operands (SassMemory, Negated Regs): return as String Literal
    return cst.SimpleString(f"'{raw}'")

  def _make_call(self, opcode: str, args: List[cst.Arg]) -> cst.Call:
    """Constructs a `sass.OPCODE(...)` function call in Python CST.

    Args:
        opcode (str): The name of the SASS operation.
        args (List[cst.Arg]): The arguments to pass to the function call.

    Returns:
        cst.Call: The constructed LibCST Call expression.
    """
    return cst.Call(func=cst.Attribute(value=cst.Name("sass"), attr=cst.Name(opcode)), args=args)
