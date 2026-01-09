"""
Signature Rewriting Logic.

Handles argument injection and stripping.
"""

from typing import Optional, TYPE_CHECKING
import libcst as cst

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.structure import StructureStage


class FuncSignatureMixin:
  """
  Mixin for modifying function signatures.
  Expecting host to provide _create_dotted_name.
  """

  def _inject_argument_to_signature(
    self: "StructureStage", node: cst.FunctionDef, arg_name: str, annotation: Optional[str] = None
  ) -> cst.FunctionDef:
    """
    Injects typed argument into signature.
    """
    params = list(node.params.params)

    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    anno_node = None
    if annotation:
      # Relies on StructureStage._create_dotted_name
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
    Removes argument by name.
    """
    params = list(node.params.params)
    new_params = [p for p in params if not (isinstance(p.name, cst.Name) and p.name.value == arg_name)]

    for i in range(len(new_params) - 1):
      if new_params[i].comma == cst.MaybeSentinel.DEFAULT:
        new_params[i] = new_params[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if len(new_params) > 0:
      new_params[-1] = new_params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=new_params)
    return node.with_changes(params=new_params_node)
