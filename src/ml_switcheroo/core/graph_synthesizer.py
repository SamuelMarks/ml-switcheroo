"""
Graph Synthesizer Module.

This module provides the `GraphSynthesizer` class, which converts a high-level
`LogicalGraph` representation back into executable Python source code (specifically
PyTorch `nn.Module` definitions).

It is used in the "Lifting" pipeline (Decompilation) to reconstruct model code
from intermediate graph representations extracted from SASS or other IRs.
"""

import libcst as cst
from typing import List, Dict, Any, Optional

from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, topological_sort


class GraphSynthesizer:
  """
  Synthesizes a Python CST Module from a LogicalGraph.

  Primarily targets PyTorch `nn.Module` generation.
  """

  def __init__(self, framework: str = "torch") -> None:
    """
    Initialize the synthesizer.

    Args:
        framework (str): Target framework string (default: "torch").
                         Currently primarily influences import generation.
    """
    self.framework = framework

  def generate(self, graph: LogicalGraph, class_name: str = "DecompiledNet") -> str:
    """
    Generates Python source code for the given graph.

    Constructs a standard `nn.Module` class with:
    1. An `__init__` method defining stateful layers.
    2. A `forward` method sequencing operations.

    Args:
        graph (LogicalGraph): The input computation graph.
        class_name (str): The name of the generated class.

    Returns:
        str: The formatted Python source code.
    """
    # 1. Imports
    body: List[cst.CSTNode] = [
      cst.parse_statement("import torch"),
      cst.parse_statement("import torch.nn as nn"),
      cst.EmptyLine(),
    ]

    # 2. Sort nodes for definition/execution order
    ordered_nodes = topological_sort(graph)

    # 3. Build Class Body
    init_func = self._build_init(ordered_nodes)
    forward_func = self._build_forward(ordered_nodes)

    class_def = cst.ClassDef(
      name=cst.Name(class_name),
      bases=[cst.Arg(value=cst.parse_expression("nn.Module"))],
      body=cst.IndentedBlock(
        body=[
          init_func,
          cst.EmptyLine(),
          forward_func,
        ]
      ),
    )
    body.append(class_def)

    module = cst.Module(body=body)
    return module.code

  def _build_init(self, nodes: List[LogicalNode]) -> cst.FunctionDef:
    """
    Constructs the `__init__` method.

    Args:
        nodes (List[LogicalNode]): The topologically sorted list of nodes.

    Returns:
        cst.FunctionDef: The AST node for the init method.
    """
    stmts: List[cst.BaseStatement] = [cst.parse_statement("super().__init__()")]

    for node in nodes:
      if self._is_stateful_layer(node):
        # self.{id} = nn.{Kind}(...)
        assignment = self._generate_layer_init(node)
        stmts.append(assignment)

    return cst.FunctionDef(
      name=cst.Name("__init__"),
      params=cst.Parameters(params=[cst.Param(name=cst.Name("self"))]),
      body=cst.IndentedBlock(body=stmts),
    )

  def _build_forward(self, nodes: List[LogicalNode]) -> cst.FunctionDef:
    """
    Constructs the `forward` method based on data flow.

    Args:
        nodes (List[LogicalNode]): The topologically sorted list of nodes.

    Returns:
        cst.FunctionDef: The AST node for the forward method.
    """
    stmts: List[cst.BaseStatement] = []

    # Determine Input Variable Name (Default to 'x')
    input_nodes = [n for n in nodes if n.kind == "Input"]
    input_arg_name = "x"
    if input_nodes:
      # Use metadata name if available, else node ID
      input_arg_name = input_nodes[0].metadata.get("name", input_nodes[0].id)

    # Track the "current" tensor variable name flowing through the graph
    current_var = input_arg_name

    for node in nodes:
      if node.kind == "Input":
        continue

      if node.kind == "Output":
        stmts.append(cst.parse_statement(f"return {current_var}"))
        continue

      # Generate Call
      if self._is_stateful_layer(node):
        # x = self.layer(x)
        line = f"{current_var} = self.{node.id}({current_var})"
        stmts.append(cst.parse_statement(line))
      else:
        # Functional op: x = torch.flatten(x)
        # If kind contains dot (torch.flatten), use it. Else assume functional?
        # The lifter produces dotted names for unmapped ops.
        func_api = node.kind
        # Generate args string (input + metadata args if any)
        args_str = current_var
        if node.metadata:
          extra_args = self._format_args_from_metadata(node.metadata)
          if extra_args:
            args_str += f", {extra_args}"

        line = f"{current_var} = {func_api}({args_str})"
        stmts.append(cst.parse_statement(line))

    return cst.FunctionDef(
      name=cst.Name("forward"),
      params=cst.Parameters(
        params=[
          cst.Param(name=cst.Name("self")),
          cst.Param(name=cst.Name(input_arg_name)),
        ]
      ),
      body=cst.IndentedBlock(body=stmts),
    )

  def _is_stateful_layer(self, node: LogicalNode) -> bool:
    """
    Heuristic to determine if a node represents a Class-based Layer.

    Args:
        node (LogicalNode): The node to check.

    Returns:
        bool: True if stateful.
    """
    if node.kind in ["Input", "Output"]:
      return False
    # If it contains specific dots (torch.flatten), it's functional.
    # If capitalized (Conv2d) or simple, assume stateful Layer.
    if "." in node.kind and not node.kind.startswith("nn."):
      return False
    return True

  def _generate_layer_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
    """
    Generates the instantiation statement for a layer.

    Args:
        node (LogicalNode): The layer node.

    Returns:
        cst.SimpleStatementLine: Statement `self.name = nn.Kind(...)`.
    """
    kind = node.kind
    # Normalize kind to nn.{Kind} if not present
    if not kind.startswith("nn.") and not "." in kind:
      kind = f"nn.{kind}"

    args_str = self._format_args_from_metadata(node.metadata)

    code = f"self.{node.id} = {kind}({args_str})"
    return cst.parse_statement(code)

  def _format_args_from_metadata(self, metadata: Dict[str, Any]) -> str:
    """
    Formats a dictionary of metadata into an argument string.

    Args:
        metadata (Dict[str, Any]): Arguments. Keys starting with 'arg_' are positional.

    Returns:
        str: Comma-separated arguments string (e.g. "32, kernel_size=3").
    """
    if not metadata:
      return ""

    args_list = []

    # Sort by keys to handle arg_0, arg_1 order if present
    for key in sorted(metadata.keys()):
      val = str(metadata[key])
      if key.startswith("arg_"):
        args_list.append(val)
      else:
        args_list.append(f"{key}={val}")

    return ", ".join(args_list)
