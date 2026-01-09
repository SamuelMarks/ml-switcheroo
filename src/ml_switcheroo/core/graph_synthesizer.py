"""
Graph Synthesizer Module.

This module provides the `GraphSynthesizer` class, which converts a high-level
`LogicalGraph` representation back into executable Python source code.

It supports two modes:
1.  **Fresh Generation**: Creates a new Python module from scratch (for simple decompilation).
2.  **Context Preservation**: Injects the synthesized graph logic into an existing
    source tree (AST), replacing only the `__init__` and `forward` methods while
    preserving class definitions, docstrings, decorators, and auxiliary methods.
"""

from typing import Any, Dict, List, Optional, Union

import libcst as cst
from libcst import matchers as m

from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, topological_sort


class ClassBodyReplacer(cst.CSTTransformer):
  """
  Transformer to swap __init__ and forward methods in a target class.

  Attributes:
      target_class (str): The name of the class to update.
      new_init (cst.FunctionDef): The new AST node for __init__.
      new_forward (cst.FunctionDef): The new AST node for forward.
      found (bool): Flag indicating if the class was found and updated.
  """

  def __init__(
    self,
    target_class: str,
    new_init: cst.FunctionDef,
    new_forward: cst.FunctionDef,
  ) -> None:
    """
    Initialize the replacer.

    Args:
        target_class (str): Name of the class to modify.
        new_init (cst.FunctionDef): Synthesized __init__ method.
        new_forward (cst.FunctionDef): Synthesized forward method.
    """
    self.target_class = target_class
    self.new_init = new_init
    self.new_forward = new_forward
    self.found = False

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    """
    Visits the ClassDefinition and replaces methods if name matches.

    Args:
        original_node (cst.ClassDef): The original node.
        updated_node (cst.ClassDef): The node with processed children.

    Returns:
        cst.ClassDef: The modified class definition.
    """
    if original_node.name.value == self.target_class:
      self.found = True
      new_body_stmts = []

      # Map of updated methods
      replacements = {
        "__init__": self.new_init,
        "forward": self.new_forward,
        "call": self.new_forward,
        "__call__": self.new_forward,
      }

      # Track which methods we have injected to avoid duplication
      injected = set()

      # Preserve parts of the class that are NOT the methods we are replacing
      for stmt in updated_node.body.body:
        if isinstance(stmt, cst.FunctionDef):
          fname = stmt.name.value
          if fname in replacements:
            # This is a target method. Replace it.
            repl_node = replacements[fname]
            # Preserve docstring if available in original but not in new
            # (Though our synthesis doesn't generate docs yet, this is safe)
            new_body_stmts.append(repl_node)
            injected.add(fname)
          else:
            # Preserve other methods (e.g. training_step)
            new_body_stmts.append(stmt)
        else:
          # Preserve class attributes, docstrings, etc.
          new_body_stmts.append(stmt)

      # If the class didn't have the methods (e.g. partial definition), append them
      # We enforce standard PyTorch naming 'forward' for synthesis
      if "__init__" not in injected:
        new_body_stmts.append(self.new_init)

      # Check inference method presence. If we didn't replace forward/call/__call__, append 'forward'
      inference_hit = any(x in injected for x in ["forward", "call", "__call__"])
      if not inference_hit:
        new_body_stmts.append(self.new_forward)

      return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body_stmts))

    return updated_node


class GraphSynthesizer:
  """
  Synthesizes a Python CST Module from a LogicalGraph.

  Primarily targets PyTorch `nn.Module` generation structure.
  """

  def __init__(self, framework: str = "torch") -> None:
    """
    Initialize the synthesizer.

    Args:
        framework (str): Target framework string (default: "torch").
                         Influences import generation and syntax style.
    """
    self.framework = framework

  def generate(
    self,
    graph: LogicalGraph,
    class_name: str = "SwitcherooNet",
    original_tree: Optional[cst.Module] = None,
  ) -> str:
    """
    Generates Python source code for the given graph.

    If `original_tree` is provided, it attempts to verify and patch the
    existing class definition within that tree, preserving comments and
    auxiliary methods. Otherwise, it generates a fresh file.

    Args:
        graph (LogicalGraph): The input computation graph.
        class_name (str): The name of the generated class.
        original_tree (Optional[cst.Module]): The original AST to patch.

    Returns:
        str: The formatted Python source code.
    """
    # 1. Sort nodes for definition/execution order
    ordered_nodes = topological_sort(graph)

    # 2. Build Method Bodies (Fresh)
    init_func = self._build_init(ordered_nodes)
    forward_func = self._build_forward(ordered_nodes)

    # 3. Strategy: Patch or Create
    if original_tree:
      # Context Preservation Mode
      replacer = ClassBodyReplacer(class_name, init_func, forward_func)
      new_tree = original_tree.visit(replacer)

      if replacer.found:
        return new_tree.code

      # Fallback: If class not found in original tree, append it?
      # Or assume the user wants clean gen.
      # We fall through to clean gen but log warning in a real system.
      pass

    # Fresh Mode / Fallback
    body: List[cst.CSTNode] = [
      cst.parse_statement("import torch"),
      cst.parse_statement("import torch.nn as nn"),
      cst.EmptyLine(),
    ]

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

    # Determine if we have inputs that are not 'x' to add to args?
    # Typically models are initialized with hyperparameters, but LogicalGraph
    # doesn't strictly track init-args yet. We assume strict default init.

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
        # Functional op: x = F.relu(x)
        func_api = node.kind
        # Heuristic: If it looks like a class (Conv2d), assume it's meant to be functional?
        # No, GraphOptimizer replaces fused nodes with Macros usually.

        # Generate args string (input + metadata args if any)
        args_str = current_var
        if node.metadata:
          extra_args = self._format_args_from_metadata(node.metadata)
          if extra_args:
            args_str += f", {extra_args}"

        line = f"{current_var} = {func_api}({args_str})"
        stmts.append(cst.parse_statement(line))

    # If last stmt wasn't return, add one
    if not stmts or not m.matches(stmts[-1], m.Return()):
      stmts.append(cst.parse_statement(f"return {current_var}"))

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
    if not kind.startswith("nn.") and "." not in kind:
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
