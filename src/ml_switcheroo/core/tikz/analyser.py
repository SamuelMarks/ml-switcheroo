"""
Static Graph Extractor for TikZ Visualization.

This module implements the analysis logic to convert Python source code (AST)
into a language-agnostic Logical Graph. It parses `__init__` methods to identify
layer definitions (Nodes) and `forward`/`__call__` methods to identify variable
data flow (Edges).

The output `LogicalGraph` is an intermediate representation used by the TikZ
emitter to generate visual diagrams.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
import libcst as cst
from libcst import matchers as m

from ml_switcheroo.core.scanners import get_full_name


@dataclass
class LogicalNode:
  """
  Represents a computation unit (Layer) in the graph.

  Attributes:
      id: Unique identifier (e.g. 'conv1').
      kind: Operation type (e.g. 'Conv2d', 'Input', 'Output').
      metadata: Dictionary of configuration parameters (e.g. kernel_size=3).
  """

  id: str
  kind: str
  metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class LogicalEdge:
  """
  Represents data flow between two nodes.
  """

  source: str
  target: str


@dataclass
class LogicalGraph:
  """
  Language-agnostic representation of the neural network structure.
  """

  nodes: List[LogicalNode] = field(default_factory=list)
  edges: List[LogicalEdge] = field(default_factory=list)


class GraphExtractor(cst.CSTVisitor):
  """
  LibCST Visitor that extracts a LogicalGraph from Python source code.

  Two-Pass Logic:
  1.  **Init Pass**: Scans `__init__` or `setup` to register named layers
      assigned to `self`. Populates the node registry.
  2.  **Forward Pass**: Scans `forward` or `__call__` to trace variable usage.
      Builds edges between registered nodes based on data flow.
  """

  def __init__(self):
    self.graph = LogicalGraph()

    # State Tracking
    self.layer_registry: Dict[str, LogicalNode] = {}  # attr_name -> Node
    self.provenance: Dict[str, str] = {}  # var_name -> node_id_that_produced_it

    self._in_init = False
    self._in_forward = False
    self._forward_args: List[str] = []

  def leave_Module(self, original_node: cst.Module) -> None:
    """
    Finalize graph construction after visiting the whole module.
    This ensures nodes are populated even if there is no forward pass.
    """
    self._finalize_graph()

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """
    Detects entry into lifecycle methods (__init__, forward, etc).
    """
    name = node.name.value
    if name in ["__init__", "setup"]:
      self._in_init = True
    elif name in ["forward", "__call__", "call"]:
      self._in_init = False  # Safety reset
      self._in_forward = True
      # Reset provenance for new forward pass analysis
      self.provenance = {}
      self._extract_input_args(node)

    return True

  def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
    """
    Resets context flags upon exiting methods.
    """
    if self._in_init:
      self._in_init = False
    elif self._in_forward:
      self._in_forward = False

  def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
    """
    Handles assignment logic for both layer definition and data flow.
    """
    if self._in_init:
      self._analyze_layer_def(node)
    elif self._in_forward:
      self._analyze_data_flow(node)
    return True

  def visit_Return(self, node: cst.Return) -> Optional[bool]:
    """
    Handles return statements in forward pass to create Output nodes.
    Also handles case where return contains a functional call directly.
    """
    if self._in_forward and node.value:
      # 1. Check if returning a direct call (e.g. return self.layer(x))
      if isinstance(node.value, cst.Call):
        # Analyze the call to generate edges, assigning implicit result to 'output'
        self._analyze_call_expression(node.value, output_vars=[])
        # The _analyze_call_expression logic creates edges to the *layer*.
        # We need to link that layer to Output.
        # Since we don't have a variable, we infer from layer logic?
        # Actually, capturing the var provenance is tricky here.
        # Let's see if we can resolve the layer name.
        layer_name = self._resolve_layer_or_func_name(node.value.func)
        if layer_name:
          # Link layer -> Output
          out_id = "output"
          if out_id not in self.layer_registry:
            self.layer_registry[out_id] = LogicalNode(out_id, "Output", {})
          self.graph.edges.append(LogicalEdge(layer_name, out_id))
        return False

      # 2. Check if returning a variable
      var_name = self._get_var_name(node.value)
      if var_name and var_name in self.provenance:
        source_id = self.provenance[var_name]
        out_id = "output"
        if out_id not in self.layer_registry:
          self.layer_registry[out_id] = LogicalNode(out_id, "Output", {})
        self.graph.edges.append(LogicalEdge(source_id, out_id))

    return False

  def _extract_input_args(self, node: cst.FunctionDef) -> None:
    """
    Registers function arguments as input sources.
    """
    # Create explicit Input node
    input_id = "input"
    self.layer_registry[input_id] = LogicalNode(input_id, "Input", {})

    for param in node.params.params:
      if param.name.value == "self":
        continue
      arg_name = param.name.value
      # Mark this variable as coming from "Input"
      self.provenance[arg_name] = input_id

  def _analyze_layer_def(self, node: cst.Assign) -> None:
    """
    Parses `self.conv = nn.Conv2d(...)` lines.
    """
    # 1. Identify Target (must be self.something)
    target = node.targets[0].target
    if not (m.matches(target, m.Attribute()) and m.matches(target.value, m.Name("self"))):
      return

    attr_name = target.attr.value

    # 2. Identify Op Type
    call = node.value
    if not isinstance(call, cst.Call):
      return

    op_type = get_full_name(call.func)
    # Simplify name (e.g. torch.nn.Conv2d -> Conv2d)
    if "." in op_type:
      op_type = op_type.split(".")[-1]

    # 3. Extract Metadata (args)
    metadata = {}
    for i, arg in enumerate(call.args):
      key = f"arg_{i}"
      val = self._node_to_string(arg.value)
      if arg.keyword:
        key = arg.keyword.value
      metadata[key] = val

    # Register
    self.layer_registry[attr_name] = LogicalNode(attr_name, op_type, metadata)

  def _analyze_data_flow(self, node: cst.Assign) -> None:
    """
    Parses `x = self.layer(x)` assignments.
    """
    # Support simple assignment: target = call
    if not isinstance(node.value, cst.Call):
      return

    # Determine output variables
    targets = []
    for target in node.targets:
      out_var_name = self._get_var_name(target.target)
      if out_var_name:
        targets.append(out_var_name)

    self._analyze_call_expression(node.value, targets)

  def _resolve_layer_or_func_name(self, func_node: cst.BaseExpression) -> Optional[str]:
    """Resolves `self.layer` -> `layer` or `F.relu` -> `func_relu`."""
    # 1. Method call on self (Registered Layer)
    if m.matches(func_node, m.Attribute()) and m.matches(func_node.value, m.Name("self")):
      return func_node.attr.value

    # 2. Functional Call (Ephemeral Node)
    # Simplified: tracing only explicit layers from init + functional if identified
    func_name = get_full_name(func_node)
    if func_name:
      # Create ad-hoc functional node
      layer_name = f"func_{func_name.split('.')[-1].lower()}"
      # Register if new
      if layer_name not in self.layer_registry:
        self.layer_registry[layer_name] = LogicalNode(layer_name, func_name, {})
      return layer_name

    return None

  def _analyze_call_expression(self, call: cst.Call, output_vars: List[str]) -> None:
    """
    Common logic to trace edges from a Call usage.
    """
    layer_name = self._resolve_layer_or_func_name(call.func)

    if not layer_name:
      return

    # Trace Inputs -> This Layer
    for arg in call.args:
      var_name = self._get_var_name(arg.value)
      if var_name and var_name in self.provenance:
        source_id = self.provenance[var_name]
        self.graph.edges.append(LogicalEdge(source_id, layer_name))

    # Update output provenance
    for out_var in output_vars:
      self.provenance[out_var] = layer_name

  def _get_var_name(self, node: cst.BaseExpression) -> Optional[str]:
    """Extracts variable name if simple identifier."""
    if isinstance(node, cst.Name):
      return node.value
    return None

  def _node_to_string(self, node: cst.CSTNode) -> str:
    """Extracts simple source string representation."""
    if isinstance(node, (cst.Integer, cst.Float)):
      return node.value
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.SimpleString):
      return node.value.strip("'\"")
    return "..."

  def _finalize_graph(self) -> None:
    """Populates the graph nodes list from the registry."""
    # Convert registry dict to list
    # Filter: only include nodes that are part of edges?
    # Better: Include all registered layers, as disjoint nodes might exist.
    if self.layer_registry:
      self.graph.nodes = list(self.layer_registry.values())
