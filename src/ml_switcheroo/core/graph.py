"""
Shared Graph Intermediate Representation.

This module provides the language-agnostic data structures and extraction logic
used to convert Python ASTs into a logical graph of operations (Layers) and
data flow (Edges).

Shared by:

- TikZ Backend (Visualizer)
- HTML Backend (Grid Layout)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict, deque
import libcst as cst
from libcst import matchers as m

from ml_switcheroo.core.scanners import get_full_name
from ml_switcheroo.utils.node_diff import capture_node_source


@dataclass
class LogicalNode:
  """
  Represents a computation unit (Layer) in the graph.
  """

  id: str
  """Unique identifier (e.g. 'conv1')."""

  kind: str
  """Operation type (e.g. 'Conv2d', 'Input', 'Output')."""

  metadata: Dict[str, str] = field(default_factory=dict)
  """Dictionary of configuration parameters (e.g. ``kernel_size=3``)."""


@dataclass
class LogicalEdge:
  """
  Represents data flow between two nodes.
  """

  source: str
  """Source node ID."""

  target: str
  """Target node ID."""


@dataclass
class LogicalGraph:
  """
  Language-agnostic representation of the neural network structure.
  """

  nodes: List[LogicalNode] = field(default_factory=list)
  """Ordered list of nodes in the graph."""

  edges: List[LogicalEdge] = field(default_factory=list)
  """List of directed edges between nodes."""


def topological_sort(graph: LogicalGraph) -> List[LogicalNode]:
  """
  Sorts graph nodes by dependency order.

  Args:
      graph: The logical graph.

  Returns:
      List of nodes in execution order.
  """
  adj = defaultdict(list)
  in_degree = defaultdict(int)
  nodes_by_id = {n.id: n for n in graph.nodes}

  # Initialize in-degree
  for n in graph.nodes:
    in_degree[n.id] = 0

  for edge in graph.edges:
    adj[edge.source].append(edge.target)
    in_degree[edge.target] += 1

  # Simple queue-based toposort
  # Note: Using sorted keys for determinism in queue initialization
  initial_roots = sorted([n.id for n in graph.nodes if in_degree[n.id] == 0])
  queue = deque(initial_roots)
  sorted_nodes = []

  while queue:
    u = queue.popleft()
    if u in nodes_by_id:
      sorted_nodes.append(nodes_by_id[u])

    for v in adj[u]:
      in_degree[v] -= 1
      if in_degree[v] == 0:
        queue.append(v)

  # Handle disconnected components or cycles by appending remaining nodes
  if len(sorted_nodes) < len(graph.nodes):
    seen = {n.id for n in sorted_nodes}
    # Append remaining nodes in definition order (fallback)
    for n in graph.nodes:
      if n.id not in seen:
        sorted_nodes.append(n)

  return sorted_nodes


class GraphExtractor(cst.CSTVisitor):
  """
  LibCST Visitor that extracts a LogicalGraph from Python source code.

  Two-Pass Logic:

  1.  **Init Pass**: Scans ``__init__`` or ``setup`` to register named layers
      assigned to ``self``. Populates the node registry.
  2.  **Forward Pass**: Scans ``forward`` or ``__call__`` to trace variable usage.
      Builds edges between registered nodes based on data flow.
  """

  def __init__(self) -> None:
    """Initialize the extractor state."""
    self.graph = LogicalGraph()

    # State Tracking
    self.layer_registry: Dict[str, LogicalNode] = {}  # attr_name -> Node
    self.provenance: Dict[str, str] = {}  # var_name -> node_id_that_produced_it
    self.model_name: str = "GeneratedNet"

    self._in_init = False
    self._in_forward = False
    self._scope_depth = 0

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """
    Capture the model class name.

    Args:
        node: ClassDef node.

    Returns:
        True to visit children.
    """
    self.model_name = node.name.value
    self._scope_depth += 1
    return True

  def leave_ClassDef(self, node: cst.ClassDef) -> None:
    self._scope_depth -= 1

  def leave_Module(self, original_node: cst.Module) -> None:
    """
    Finalize graph construction after visiting the whole module.
    Populates the nodes list from the registry.

    Args:
        original_node: The module node.
    """
    self._finalize_graph()

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """
    Detects entry into lifecycle methods (__init__, forward, etc).

    Args:
        node: The function definition node.

    Returns:
        True to visit children.
    """
    name = node.name.value
    self._scope_depth += 1
    if name in ["__init__", "setup"]:
      self._in_init = True
    elif name in ["forward", "__call__", "call", "kernel", "f"]:
      self._in_init = False  # Safety reset
      self._in_forward = True
      # Reset provenance for new forward pass analysis
      self.provenance = {}
      self._extract_input_args(node)

    return True

  def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
    """
    Resets context flags upon exiting methods.

    Args:
        node: The function definition node.
    """
    self._scope_depth -= 1
    if self._in_init:
      self._in_init = False
    elif self._in_forward:
      self._in_forward = False

  def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
    """
    Handles assignment logic for both layer definition and data flow.

    Args:
        node: The assignment node.

    Returns:
        True to visit children.
    """
    if self._in_init:
      self._analyze_layer_def(node)
    elif self._in_forward:
      self._analyze_data_flow(node)
    elif self._scope_depth == 0:
      # Top-level script mode
      self._analyze_data_flow(node)
    return True

  def visit_Return(self, node: cst.Return) -> Optional[bool]:
    """
    Handles return statements in forward pass to create Output nodes.
    Links the variable returned to an implicit 'Output' node.

    Args:
        node: The return statement node.

    Returns:
        False to stop recursion into the return statement (logic handled here).
    """
    # Handle both explicit function context or implicit script context if result printed/last
    if self._in_forward and node.value:
      # 1. Check if returning a direct call (e.g. return self.layer(x))
      if isinstance(node.value, cst.Call):
        self._analyze_call_expression(node.value, output_vars=[])
        layer_name = self._resolve_layer_or_func_name(node.value.func)
        if layer_name:
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
    Creates distinct Input nodes for each argument to allow SASS register mapping.

    Args:
        node: The function definition.
    """
    for param in node.params.params:
      if param.name.value == "self":
        continue
      arg_name = param.name.value
      # Unique node ID for this input
      input_id = f"Input_{arg_name}"  # e.g. "Input_x"

      # Register node if not exists
      if input_id not in self.layer_registry:
        self.layer_registry[input_id] = LogicalNode(input_id, "Input", {"name": arg_name})

      # Map valid variable name to this input node
      self.provenance[arg_name] = input_id

  def _analyze_layer_def(self, node: cst.Assign) -> None:
    """
    Parses ``self.conv = nn.Conv2d(...)`` lines in ``__init__``.

    Args:
        node: The assignment statement.
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
      val = capture_node_source(arg.value)
      if arg.keyword:
        key = arg.keyword.value
      metadata[key] = val

    # Register
    self.layer_registry[attr_name] = LogicalNode(attr_name, op_type, metadata)

  def _analyze_data_flow(self, node: cst.Assign) -> None:
    """
    Parses ``x = self.layer(x)`` assignments in forward.

    Args:
        node: The assignment statement.
    """
    if not isinstance(node.value, cst.Call):
      return

    targets = []
    for target in node.targets:
      out_var_name = self._get_var_name(target.target)
      if out_var_name:
        targets.append(out_var_name)

    self._analyze_call_expression(node.value, targets)

  def _resolve_layer_or_func_name(self, func_node: cst.BaseExpression) -> Optional[str]:
    """
    Resolves ``self.layer`` -> ``layer`` or ``F.relu`` -> ``func_relu``.

    Args:
        func_node: The function expression node inside the call.

    Returns:
        The resolved identifier string or None.
    """
    # 1. Method call on self (Registered Layer)
    if m.matches(func_node, m.Attribute()) and m.matches(func_node.value, m.Name("self")):
      return func_node.attr.value

    # 2. Functional Call (Ephemeral Node)
    # Create ad-hoc node if functional (e.g. F.relu)
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

    Args:
        call: The call expression.
        output_vars: List of variable names receiving the result.
    """
    layer_name = self._resolve_layer_or_func_name(call.func)

    if not layer_name:
      return

    # Trace Inputs -> This Layer
    for arg in call.args:
      var_name = self._get_var_name(arg.value)

      # Handle implicit external input (script mode)
      if var_name and var_name not in self.provenance:
        if self._scope_depth == 0:
          ext_id = f"Input_{var_name}"
          if ext_id not in self.layer_registry:
            self.layer_registry[ext_id] = LogicalNode(ext_id, "Input", {"name": var_name})
          self.provenance[var_name] = ext_id

      if var_name and var_name in self.provenance:
        source_id = self.provenance[var_name]
        self.graph.edges.append(LogicalEdge(source_id, layer_name))

    # Update output provenance
    for out_var in output_vars:
      self.provenance[out_var] = layer_name

  def _get_var_name(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Extracts variable name if simple identifier.

    Args:
        node: The expression node.

    Returns:
        Variable name or None.
    """
    if isinstance(node, cst.Name):
      return node.value
    return None

  def _finalize_graph(self) -> None:
    """Populates the graph nodes list from the registry."""
    if self.layer_registry:
      self.graph.nodes = list(self.layer_registry.values())
