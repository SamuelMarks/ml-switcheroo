"""Python Snippet Emitter.

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

import typing

import libcst as cst
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.exceptions import BackendEmissionError, SyntaxParseError
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.schema import StructuralTraits


class PythonSnippetEmitter:
  """Generate isolated Python statements from LogicalNodes.

  Adapts generation logic based on the target framework's ``StructuralTraits``
  to ensure idiomatic code (e.g., injecting `rngs` when required, resolving layer
  and functional namespace prefixes).
  """

  def __init__(self, framework: str = "torch") -> None:
    """Initialize the snippet emitter.

    Args:
        framework: The target framework key (e.g., 'torch', 'jax', 'flax_nnx').

    """
    self.framework = framework
    adapter = get_adapter(framework)
    self.traits: StructuralTraits = adapter.structural_traits if adapter is not None else StructuralTraits()

  def emit_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
    """Generate the initialization statement for a stateful layer.

    Args:
        node: The logical node describing the layer.

    Returns:
        A LibCST statement node representing `self.id = Kind(...)`.

    """
    if not self._is_stateful_layer(node):
      return cst.SimpleStatementLine(body=[cst.Pass()])

    kind = self._resolve_api_name(node.op_type)
    func_node = cst.parse_expression(kind)

    args_list = self._build_args_from_metadata(node.attributes)

    if self.traits.requires_rng_in_init:
      # Needs rngs=rngs, inject it directly if it doesn't already exist.
      if not any(isinstance(a.keyword, cst.Name) and a.keyword.value == "rngs" for a in args_list):
        args_list.append(cst.Arg(keyword=cst.Name("rngs"), equal=cst.AssignEqual(), value=cst.Name("rngs")))

    call_node = cst.Call(func=func_node, args=args_list)
    assign_target = cst.AssignTarget(target=cst.Attribute(value=cst.Name("self"), attr=cst.Name(node.id)))
    assign = cst.Assign(targets=[assign_target], value=call_node)

    return cst.SimpleStatementLine(body=[assign])

  def emit_call(
    self,
    node: LogicalNode,
    input_vars: list[str],
    output_var: str,
  ) -> cst.SimpleStatementLine:
    """Generate the execution call statement (Assignment).

    Args:
        node: The logical node.
        input_vars: List of variable names to pass as arguments.
        output_var: Variable name to assign the result to.

    Returns:
        A LibCST statement node (Assign).

    """
    if node.op_type == "Input":
      if input_vars and input_vars[0] != output_var:
        return cst.SimpleStatementLine(
          body=[cst.Assign(targets=[cst.AssignTarget(target=cst.Name(output_var))], value=cst.Name(input_vars[0]))]
        )
      return cst.SimpleStatementLine(body=[cst.Pass()])

    call_expr = self.emit_expression(node, input_vars)

    return cst.SimpleStatementLine(
      body=[cst.Assign(targets=[cst.AssignTarget(target=cst.Name(output_var))], value=call_expr)]
    )

  def emit_expression(self, node: LogicalNode, input_vars: list[str]) -> cst.BaseExpression:
    """Generate the function call expression (without assignment).

    Args:
        node: The logical node.
        input_vars: List of input variable names.

    Returns:
        A LibCST expression node.

    """
    try:
      func_node: cst.BaseExpression
      if self._is_stateful_layer(node):
        func_node = cst.Attribute(value=cst.Name("self"), attr=cst.Name(node.id))
      else:
        func_name = self._resolve_api_name(node.op_type)
        func_node = cst.parse_expression(func_name)

      args_list = []
      for v in input_vars:
        args_list.append(cst.Arg(value=cst.parse_expression(v)))

      attrs = node.attributes
      if not self._is_stateful_layer(node) and attrs:
        extra_args = self._build_args_from_metadata(attrs)
        args_list.extend(extra_args)

      return cst.Call(func=func_node, args=args_list)
    except (cst.ParserSyntaxError, SyntaxError, ValueError, TypeError, BackendEmissionError, SyntaxParseError):
      return cst.Name("None")

  def _is_stateful_layer(self, node: LogicalNode) -> bool:
    """Determine if a logical node represents a stateful layer.

    Args:
        node: The logical node to check.

    Returns:
        True if stateful, False otherwise.
    """
    op_type = node.op_type
    if op_type in ["Input", "Output"]:
      return False
    if op_type.startswith("func_") or "functional" in op_type or "ops" in op_type:
      return False
    leaf = op_type.split(".")[-1]
    if leaf and leaf[0].isupper():
      return True
    return False

  def _resolve_api_name(self, kind: str) -> str:
    """Resolve the API name based on the framework.

    Args:
        kind: The original kind string.

    Returns:
        The resolved fully qualified API string.
    """
    if "." in kind:
      return kind

    clean_kind = kind
    if kind.startswith("func_"):
      clean_kind = kind[5:]

    layer_pref = self.traits.layer_namespace_prefix
    func_pref = self.traits.functional_namespace_prefix
    if layer_pref and clean_kind[0].isupper():
      return f"{layer_pref}.{clean_kind}"
    if func_pref and not clean_kind[0].isupper():
      return f"{func_pref}.{clean_kind}"

    return clean_kind

  def _build_args_from_metadata(self, metadata: typing.Mapping[str, AttributeValue] | None) -> list[cst.Arg]:
    """Build libcst arguments from node metadata.

    Args:
        metadata: The dictionary of metadata values.

    Returns:
        A list of libcst Arg nodes.
    """
    if not metadata:
      return []
    args_list = []
    for key in sorted(metadata.keys()):
      val = str(metadata[key])
      val_node = cst.parse_expression(val)
      if key.startswith("arg_"):
        args_list.append(cst.Arg(value=val_node))
      else:
        args_list.append(cst.Arg(keyword=cst.Name(key), equal=cst.AssignEqual(), value=val_node))
    return args_list
