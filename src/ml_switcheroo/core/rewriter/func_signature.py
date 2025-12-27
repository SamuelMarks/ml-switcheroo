"""
Signature Rewriting Logic for Function Definitions.

This module provides the `FuncSignatureMixin` used by the `structure_func` rewriter.
It handles the injection and stripping of arguments in function definitions.
"""

from typing import Optional
import libcst as cst


class FuncSignatureMixin:
  """
  Mixin for modifying function signatures.

  Assumes the host class provides:
  - ``_create_dotted_name(str) -> cst.BaseExpression`` (from BaseRewriter).
  """

  def _inject_argument_to_signature(
    self, node: cst.FunctionDef, arg_name: str, annotation: Optional[str] = None
  ) -> cst.FunctionDef:
    """
    Injects a typed argument into the function signature.
    Inserts after ``self`` if present, otherwise at index 0.

    Args:
        node: The function definition.
        arg_name: The name of the argument to inject.
        annotation: Optional type hint string (dotted notation supported).

    Returns:
        The modified function definition.
    """
    params = list(node.params.params)

    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    anno_node = None
    if annotation:
      # Note: _create_dotted_name is required from the BaseRewriter context
      anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation))

    new_param = cst.Param(name=cst.Name(arg_name), annotation=anno_node, comma=cst.MaybeSentinel.DEFAULT)
    params.insert(insert_idx, new_param)

    # Fix commas
    for i in range(len(params) - 1):
      if params[i].comma == cst.MaybeSentinel.DEFAULT:
        params[i] = params[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if len(params) > 0:
      last = params[-1]
      if last.comma != cst.MaybeSentinel.DEFAULT:
        params[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)

  def _strip_argument_from_signature(self, node: cst.FunctionDef, arg_name: str) -> cst.FunctionDef:
    """
    Removes an argument by name from the function signature.

    Args:
        node: The function definition.
        arg_name: The argument name to strip.

    Returns:
        The modified function definition.
    """
    params = list(node.params.params)
    new_params = [p for p in params if not (isinstance(p.name, cst.Name) and p.name.value == arg_name)]

    # Fix commas
    for i in range(len(new_params) - 1):
      if new_params[i].comma == cst.MaybeSentinel.DEFAULT:
        new_params[i] = new_params[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if len(new_params) > 0:
      new_params[-1] = new_params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=new_params)
    return node.with_changes(params=new_params_node)
