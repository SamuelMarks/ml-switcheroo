"""
TikZ-to-Python Synthesizer.

This module consumes a ``LogicalGraph`` (representing the layout and metadata
of a neural network extracted from TikZ) and generates valid, executable
Python source code using LibCST.

It performs the inverse of the ``analyser.py``:
1.  **Topological Sort**: Determines the execution order of the graph.
2.  **Init Generation**: Constructs layer definitions (e.g., ``self.conv = ...``)
    based on the target framework's API conventions.
3.  **Forward Generation**: Reconstructs the data flow of the ``forward`` method,
    handling variable assignment and return values.
"""

import libcst as cst
from collections import defaultdict, deque
from typing import Dict, List, Any, Set

from ml_switcheroo.core.tikz.analyser import LogicalGraph, LogicalNode, LogicalEdge


class GraphSynthesizer:
  """
  Converts a LogicalGraph into a Python Class definition.
  """

  def __init__(self, framework: str = "torch"):
    """
    Args:
        framework: Target framework ('torch' or 'jax').
                   Defaults to 'torch' for generated formatting.
    """
    self.framework = framework

  def generate(self, graph: LogicalGraph, class_name: str = "GeneratedNet") -> str:
    """
    Generates the full Python source code for the graph.

    Args:
        graph: The logical representation of the network.
        class_name: The name of the generated class.

    Returns:
        str: Python source code.
    """
    # 0. Sort nodes for execution order
    ordered_nodes = self._topological_sort(graph)

    # 1. Create Class Header
    base_class = "nn.Module" if self.framework == "torch" else "nnx.Module"

    # 2. Generate __init__
    init_func = self._generate_init(graph.nodes)

    # 3. Generate forward/__call__
    # Default input arg name is 'x' unless Input node specifies otherwise
    forward_func = self._generate_forward(graph, ordered_nodes)

    # 4. Assemble Module
    body_stmts = [init_func, forward_func]

    class_def = cst.ClassDef(
      name=cst.Name(class_name),
      bases=[cst.Arg(value=cst.parse_expression(base_class))],
      body=cst.IndentedBlock(body=body_stmts),
    )

    # Wrap in imports for validity
    imports = self._generate_imports()

    module = cst.Module(body=imports + [class_def])
    return module.code

  def _generate_imports(self) -> List[cst.SimpleStatementLine]:
    """Generates framework-specific imports."""
    if self.framework == "torch":
      return [
        cst.parse_statement("import torch"),
        cst.parse_statement("import torch.nn as nn"),
      ]
    elif self.framework == "jax":
      return [
        cst.parse_statement("from flax import nnx"),
        cst.parse_statement("import jax.numpy as jnp"),
      ]
    return []

  def _generate_init(self, nodes: List[LogicalNode]) -> cst.FunctionDef:
    """
    Constructs the __init__ method.
    """
    body = []

    # Super init for Torch
    if self.framework == "torch":
      body.append(cst.parse_statement("super().__init__()"))

    # Layer definitions
    for node in nodes:
      if node.kind in ["Input", "Output"]:
        continue

      # Construct: self.id = API(config)
      lhs = cst.parse_expression(f"self.{node.id}")

      # Resolve API prefix
      prefix = "nn" if self.framework == "torch" else "nnx"
      api_call = f"{prefix}.{node.kind}"

      # Build args string
      args_parts = []
      for k, v in node.metadata.items():
        # Handle positional args coming from generic metadata keys like 'arg_0'
        if k.startswith("arg_"):
          args_parts.append(v)
        else:
          args_parts.append(f"{k}={v}")

      # In JAX (Flax NNX), inject rngs if it looks like a layer
      if self.framework == "jax" and "arg_0" in node.metadata:  # Heuristic for layer
        if not any("rngs" in a for a in args_parts):
          args_parts.append("rngs=rngs")

      args_str = ", ".join(args_parts)
      rhs = cst.parse_expression(f"{api_call}({args_str})")

      assign = cst.Assign(targets=[cst.AssignTarget(target=lhs)], value=rhs)
      body.append(cst.SimpleStatementLine([assign]))

    # Params
    params = [cst.Param(name=cst.Name("self"))]
    if self.framework == "jax":
      # Inject rngs argument for NNX
      params.append(cst.Param(name=cst.Name("rngs"), annotation=cst.Annotation(cst.parse_expression("nnx.Rngs"))))

    return cst.FunctionDef(
      name=cst.Name("__init__"),
      params=cst.Parameters(params=params),
      body=cst.IndentedBlock(body=body),
    )

  def _generate_forward(self, graph: LogicalGraph, ordered_nodes: List[LogicalNode]) -> cst.FunctionDef:
    """
    Constructs the forward (or __call__) method based on graph topology.
    """
    func_name = "forward" if self.framework == "torch" else "__call__"
    body = []

    # Track variable names holding the output of each node
    # node_id -> variable_name
    var_map = {}

    # Build Adjacency for inputs
    # node_id -> list of source_node_ids
    inputs_map = defaultdict(list)
    for edge in graph.edges:
      inputs_map[edge.target].append(edge.source)

    # Handle explicit Input node arguments
    input_node = next((n for n in ordered_nodes if n.kind == "Input"), None)
    input_var = "x"
    if input_node:
      var_map[input_node.id] = input_var

    # If no input node found, assume 'x' is the input arg implicitly
    params = [cst.Param(name=cst.Name("self"))]
    params.append(cst.Param(name=cst.Name(input_var)))

    last_var = input_var

    for node in ordered_nodes:
      if node.kind == "Input":
        continue

      if node.kind == "Output":
        # Find source feeding output
        sources = inputs_map[node.id]
        if sources:
          # Return the result of the source
          ret_val = var_map.get(sources[0], last_var)
          body.append(cst.parse_statement(f"return {ret_val}"))
        continue

      # Determine input variables for this node
      sources = inputs_map[node.id]
      if not sources:
        # Disconnected node or uses default input logic
        # Only warn/skip if not reached effectively?
        # We assume sequential default if not wired.
        inputs = [last_var]
      else:
        inputs = [var_map.get(s, "unknown") for s in sources]

      # Construct call: x = self.layer(input)
      # Or if multi-input: x = self.layer(a, b)
      call_args = ", ".join(inputs)

      # Result variable is usually 'x' if chain, or unique if branching
      # Simple heuristic: Always overwrite x for linear, use unique for branching
      # For robustness: Always use node_id as var name unless it's a simple chain?
      # Let's reuse 'x' if only 1 input and this is the only consumer?
      # Safe strategy: Assign to node_id.
      out_var = node.id
      var_map[node.id] = out_var
      last_var = out_var

      stmt = cst.parse_statement(f"{out_var} = self.{node.id}({call_args})")
      body.append(stmt)

    # Ensure return if no Output node
    if not any(n.kind == "Output" for n in ordered_nodes) and body:
      body.append(cst.parse_statement(f"return {last_var}"))

    return cst.FunctionDef(
      name=cst.Name(func_name),
      params=cst.Parameters(params=params),
      body=cst.IndentedBlock(body=body),
    )

  def _topological_sort(self, graph: LogicalGraph) -> List[LogicalNode]:
    """
    Sorts nodes by dependency order.
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

    queue = deque([n.id for n in graph.nodes if in_degree[n.id] == 0])
    sorted_nodes = []

    while queue:
      u = queue.popleft()
      if u in nodes_by_id:
        sorted_nodes.append(nodes_by_id[u])

      for v in adj[u]:
        in_degree[v] -= 1
        if in_degree[v] == 0:
          queue.append(v)

    # Append disjoint nodes (fallback)
    if len(sorted_nodes) < len(graph.nodes):
      seen = {n.id for n in sorted_nodes}
      for n in graph.nodes:
        if n.id not in seen:
          sorted_nodes.append(n)

    return sorted_nodes
