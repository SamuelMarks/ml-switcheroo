"""
Graph Extraction Frontend.

This module is responsible for analyzing Python Abstract Syntax Trees (ASTs) using LibCST
and extracting a `LogicalGraph` Intermediate Representation via the `GraphExtractor`.

It performs Provenance Tracking, mapping logical nodes back to their source CST nodes,
enabling surgical patching later in the pipeline.
"""

from typing import Dict, List, Optional, Any, Union
import libcst as cst
from libcst import matchers as m

# Re-export Core IR definitions for backward compatibility
from ml_switcheroo.compiler.ir import LogicalNode, LogicalEdge, LogicalGraph, topological_sort
from ml_switcheroo.core.scanners import get_full_name
from ml_switcheroo.utils.node_diff import capture_node_source

__all__ = ["LogicalNode", "LogicalEdge", "LogicalGraph", "topological_sort", "GraphExtractor"]


class GraphExtractor(cst.CSTVisitor):
  """
  LibCST Visitor that extracts a LogicalGraph from Python source code.

  Two-Pass Logic:
  1.  **Init Pass**: Scans ``__init__`` or ``setup`` to register named layers
      assigned to ``self``. Populates the node registry and provenance map.
  2.  **Forward Pass**: Scans ``forward`` or ``__call__`` to trace variable usage.
      Builds edges between registered nodes based on data flow.

  Attributes:
      graph (LogicalGraph): The constructed intermediate representation.
      layer_registry (Dict[str, LogicalNode]): Mapping of node IDs to LogicalNodes.
      provenance (Dict[str, str]): Mapping of variable names to producer node IDs.
      node_map (Dict[str, cst.CSTNode]): Provenance registry mapping Node ID -> CST Node.
  """

  def __init__(self) -> None:
    """Initialize the extractor state."""
    self.graph = LogicalGraph()

    # State Tracking
    self.layer_registry: Dict[str, LogicalNode] = {}  # attr_name -> Node
    self.provenance: Dict[str, str] = {}  # var_name -> node_id_that_produced_it
    self.node_map: Dict[str, cst.CSTNode] = {}
    self.model_name: str = "GeneratedNet"

    self._in_init = False
    self._in_forward = False
    self._scope_depth = 0

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """Capture the model class name."""
    self.model_name = node.name.value
    self._scope_depth += 1
    return True

  def leave_ClassDef(self, node: cst.ClassDef) -> None:
    """Exit class scope."""
    self._scope_depth -= 1

  def leave_Module(self, original_node: cst.Module) -> None:
    """Finalize graph construction."""
    self._finalize_graph()

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Detects entry into lifecycle methods."""
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
    """Resets context flags upon exiting methods."""
    self._scope_depth -= 1
    if self._in_init:
      self._in_init = False
    elif self._in_forward:
      self._in_forward = False

  def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
    """Handles assignment logic for both layer definition and data flow."""
    if self._in_init:
      self._analyze_layer_def(node)
    elif self._in_forward:
      self._analyze_data_flow(node)
    elif self._scope_depth == 0:
      self._analyze_data_flow(node)
    return True

  def visit_Return(self, node: cst.Return) -> Optional[bool]:
    """Handles return statements to identify Output nodes."""
    if self._in_forward and node.value:
      # 1. Check if returning a direct call
      if isinstance(node.value, cst.Call):
        self._analyze_call_expression(node.value, output_vars=[])
        # The inner call registered a node. Link it to output.
        layer_name = self._resolve_layer_or_func_name(node.value.func, context_node=node.value)
        if layer_name:
          out_id = "output"
          if out_id not in self.layer_registry:
            self.layer_registry[out_id] = LogicalNode(out_id, "Output", {})
            self.node_map[out_id] = node
          self.graph.edges.append(LogicalEdge(layer_name, out_id))
        return False

      # 2. Check if returning a variable
      var_name = self._get_var_name(node.value)
      if var_name and var_name in self.provenance:
        source_id = self.provenance[var_name]
        out_id = "output"
        if out_id not in self.layer_registry:
          self.layer_registry[out_id] = LogicalNode(out_id, "Output", {})
          self.node_map[out_id] = node
        self.graph.edges.append(LogicalEdge(source_id, out_id))

    return False

  # --- Extraction Helpers ---

  def _extract_input_args(self, node: cst.FunctionDef) -> None:
    """Registers function arguments as input sources."""
    for param in node.params.params:
      if param.name.value == "self":
        continue
      arg_name = param.name.value
      # Note: We use unique IDs for inputs to distinguish
      input_id = f"Input_{arg_name}"

      if input_id not in self.layer_registry:
        self.layer_registry[input_id] = LogicalNode(input_id, "Input", {"name": arg_name})
        # Provenance: The Param definition
        self.node_map[input_id] = param

      self.provenance[arg_name] = input_id

  def _analyze_layer_def(self, node: cst.Assign) -> None:
    """Parses self.layer = ... lines."""
    target = node.targets[0].target
    if not (m.matches(target, m.Attribute()) and m.matches(target.value, m.Name("self"))):
      return

    attr_name = target.attr.value
    call = node.value
    if not isinstance(call, cst.Call):
      return

    op_type = get_full_name(call.func)
    if "." in op_type:
      op_type = op_type.split(".")[-1]

    metadata = {}
    for i, arg in enumerate(call.args):
      key = f"arg_{i}"
      val = capture_node_source(arg.value)
      if arg.keyword:
        key = arg.keyword.value
      metadata[key] = val

    self.layer_registry[attr_name] = LogicalNode(attr_name, op_type, metadata)
    # Provenance: The Assign statement
    self.node_map[attr_name] = node

  def _analyze_data_flow(self, node: cst.Assign) -> None:
    """Parses x = self.layer(x) logic."""
    if self._scope_depth == 0 and isinstance(node.value, (cst.Integer, cst.Float, cst.Name)):
      for target in node.targets:
        var_name = self._get_var_name(target.target)
        if var_name:
          input_id = f"Input_{var_name}"
          if input_id not in self.layer_registry:
            val_str = capture_node_source(node.value)
            self.layer_registry[input_id] = LogicalNode(input_id, "Input", {"name": var_name, "value": val_str})
            self.provenance[var_name] = input_id
            self.node_map[input_id] = node
      return

    if not isinstance(node.value, cst.Call):
      return

    targets = []
    for target in node.targets:
      out_var_name = self._get_var_name(target.target)
      if out_var_name:
        targets.append(out_var_name)

    # For data flow assignments, we pass the Assign statement as context
    # so that if a functional op (F.relu) is created, it maps to this line.
    self._analyze_call_expression(node.value, targets, context_node=node)

  def _resolve_layer_or_func_name(
    self, func_node: cst.BaseExpression, context_node: Optional[cst.CSTNode] = None
  ) -> Optional[str]:
    """Resolves identifier to node ID. Creates functional nodes on fly."""
    if m.matches(func_node, m.Attribute()) and m.matches(func_node.value, m.Name("self")):
      return func_node.attr.value

    func_name = get_full_name(func_node)
    if func_name:
      layer_name = f"func_{func_name.split('.')[-1].lower()}"
      if layer_name not in self.layer_registry:
        self.layer_registry[layer_name] = LogicalNode(layer_name, func_name, {})
        # Provenance: Map to the call/statement that triggered creation
        if context_node:
          self.node_map[layer_name] = context_node
      return layer_name

    return None

  def _analyze_call_expression(
    self, call: cst.Call, output_vars: List[str], context_node: Optional[cst.CSTNode] = None
  ) -> None:
    """Traces edges from call inputs to the layer node."""
    # Use call itself as context if no parent statement provided
    ctx = context_node if context_node else call
    layer_name = self._resolve_layer_or_func_name(call.func, context_node=ctx)

    if not layer_name:
      return

    for arg in call.args:
      var_name = self._get_var_name(arg.value)

      # Implicit external input handling
      if var_name and var_name not in self.provenance:
        if self._scope_depth == 0:
          ext_id = f"Input_{var_name}"
          if ext_id not in self.layer_registry:
            self.layer_registry[ext_id] = LogicalNode(ext_id, "Input", {"name": var_name})
            self.node_map[ext_id] = arg
          self.provenance[var_name] = ext_id

      if var_name and var_name in self.provenance:
        source_id = self.provenance[var_name]
        self.graph.edges.append(LogicalEdge(source_id, layer_name))

    for out_var in output_vars:
      self.provenance[out_var] = layer_name

  def _get_var_name(self, node: cst.BaseExpression) -> Optional[str]:
    if isinstance(node, cst.Name):
      return node.value
    return None

  def _finalize_graph(self) -> None:
    """Copies registry values to the graph object."""
    if self.layer_registry:
      # Sort by definition order implicitly via dict preservation or explicitly if desired
      self.graph.nodes = list(self.layer_registry.values())
    self.graph.name = self.model_name
