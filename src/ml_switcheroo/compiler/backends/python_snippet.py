"""
Python Snippet Emitter.

This module provides the `PythonSnippetEmitter`, a specialized backend component
designed to generate atomic Python statements (CST nodes) from LogicalNodes.

Unlike `PythonBackend` which synthesizes entire modules/classes, this emitter
focuses on generating individual:
1.  **Initialization Statements**: `self.layer = LayerClass(...)`
2.  **Execution Statements**: `output = self.layer(input)`
3.  **Expressions**: `self.layer(input)` or `func(input)`

This is used by the ``GraphPatcher`` to surgically insert new code for
nodes created during Graph Optimization (e.g. Fused Operations).
"""

from typing import Any, Dict, List, Optional
import libcst as cst

from ml_switcheroo.compiler.ir import LogicalNode


class PythonSnippetEmitter:
  """
  Generates isolated Python statements from LogicalNodes.

  Adapts generation logic based on the target framework to ensure
  idiomatic code (e.g., injecting `rngs` for Flax, `nn.` prefix for Torch).
  """

  def __init__(self, framework: str = "torch") -> None:
    """
    Initializes the snippet emitter.

    Args:
        framework: The target framework key (e.g., 'torch', 'jax', 'flax_nnx').
    """
    self.framework = framework

  def emit_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
    """
    Generates the initialization statement for a stateful layer.

    Args:
        node: The logical node describing the layer.

    Returns:
        A LibCST statement node representing `self.id = Kind(...)`.
    """
    if not self._is_stateful_layer(node):
      return cst.SimpleStatementLine(body=[cst.Pass()])

    kind = self._resolve_api_name(node.kind)
    args_str = self._format_args_from_metadata(node.metadata)

    if self.framework in ["jax", "flax", "flax_nnx"]:
      if "rngs" not in args_str:
        suffix = ", rngs=rngs" if args_str else "rngs=rngs"
        args_str += suffix

    code = f"self.{node.id} = {kind}({args_str})"
    return cst.parse_statement(code)

  def emit_call(
    self,
    node: LogicalNode,
    input_vars: List[str],
    output_var: str,
  ) -> cst.SimpleStatementLine:
    """
    Generates the execution call statement (Assignment).

    Args:
        node: The logical node.
        input_vars: List of variable names to pass as arguments.
        output_var: Variable name to assign the result to.

    Returns:
        A LibCST statement node (Assign).
    """
    if node.kind == "Input":
      if input_vars and input_vars[0] != output_var:
        return cst.parse_statement(f"{output_var} = {input_vars[0]}")
      return cst.SimpleStatementLine(body=[cst.Pass()])

    call_expr = self.emit_expression(node, input_vars)

    # We manually construct the Assign node
    return cst.SimpleStatementLine(
      body=[cst.Assign(targets=[cst.AssignTarget(target=cst.Name(output_var))], value=call_expr)]
    )

  def emit_expression(self, node: LogicalNode, input_vars: List[str]) -> cst.BaseExpression:
    """
    Generates the function call expression (without assignment).

    Args:
        node: The logical node.
        input_vars: List of input variable names.

    Returns:
        A LibCST expression node.
    """
    if self._is_stateful_layer(node):
      func_name = f"self.{node.id}"
    else:
      func_name = self._resolve_api_name(node.kind)

    args_list = list(input_vars)

    # Extra args from metadata for functional calls
    if not self._is_stateful_layer(node) and node.metadata:
      extra_args = self._format_args_from_metadata(node.metadata)
      if extra_args:
        args_list.append(extra_args)

    args_str = ", ".join(args_list)
    code = f"{func_name}({args_str})"

    try:
      return cst.parse_expression(code)
    except cst.ParserSyntaxError:
      return cst.Name("None")

  def _is_stateful_layer(self, node: LogicalNode) -> bool:
    if node.kind in ["Input", "Output"]:
      return False
    if node.kind.startswith("func_") or "functional" in node.kind or "ops" in node.kind:
      return False
    leaf = node.kind.split(".")[-1]
    if leaf and leaf[0].isupper():
      return True
    return False

  def _resolve_api_name(self, kind: str) -> str:
    if "." in kind:
      return kind

    clean_kind = kind
    if kind.startswith("func_"):
      clean_kind = kind[5:]

    if self.framework == "torch":
      if clean_kind[0].isupper():
        return f"nn.{clean_kind}"
      return f"torch.{clean_kind}"

    elif self.framework in ["jax", "flax", "flax_nnx"]:
      if clean_kind[0].isupper():
        return f"nnx.{clean_kind}"
      return f"jnp.{clean_kind}"

    elif self.framework == "keras":
      if clean_kind[0].isupper():
        return f"keras.layers.{clean_kind}"
      return f"keras.ops.{clean_kind}"

    return clean_kind

  def _format_args_from_metadata(self, metadata: Dict[str, Any]) -> str:
    if not metadata:
      return ""
    args_list = []
    for key in sorted(metadata.keys()):
      val = str(metadata[key])
      if key.startswith("arg_"):
        args_list.append(val)
      else:
        args_list.append(f"{key}={val}")
    return ", ".join(args_list)
