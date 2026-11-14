"""Plugin for Functional Dataset Pipeline Generation.

Converts generic ``DataLoader`` constructions into chained dataset pipelines
(``.from_tensor_slices(...).shuffle(...).batch(...).prefetch(...)``) driven
strictly by ``HookContext.current_variant`` without hardcoded framework paths.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name, resolve_variant_api

_DATASET_KEYWORDS = frozenset({"dataset", "tensors", "data"})


def _get_arg_by_name(args: typing.Sequence[cst.Arg], name: str) -> cst.Arg | None:
  """Retrieve an argument node by its keyword name.

  Args:
      args: Sequence of call arguments.
      name: Keyword string to search for.

  Returns:
      Matched ``cst.Arg`` or ``None``.
  """
  for arg in args:
    if arg.keyword and arg.keyword.value == name:
      return arg
  return None


def _extract_dataset_expr(args: typing.Sequence[cst.Arg]) -> cst.BaseExpression | None:
  """Extract the dataset expression from positional or keyword call arguments.

  Args:
      args: Call arguments passed to ``DataLoader(...)``.

  Returns:
      Dataset expression node if found, otherwise ``None``.
  """
  for kw in ("dataset", "tensors", "data"):
    found = _get_arg_by_name(args, kw)
    if found is not None:
      return found.value
  for arg in args:
    if arg.keyword is None and not arg.star:
      return arg.value
  return None


def _extract_tensor_dataset_inputs(node: cst.BaseExpression) -> list[cst.BaseExpression] | None:
  """Unwrap ``TensorDataset(x, y)`` wrapper calls into constituent tensor expressions.

  Args:
      node: Candidate dataset expression node.

  Returns:
      List of unwrapped tensor expressions if ``node`` calls ``TensorDataset``, else ``None``.
  """
  if isinstance(node, cst.Call):
    func_name = ""
    if isinstance(node.func, cst.Name):
      func_name = node.func.value
    elif isinstance(node.func, cst.Attribute):
      func_name = node.func.attr.value
    if func_name == "TensorDataset":
      return [arg.value for arg in node.args if not arg.star]
  return None


@register_hook("tf_data_loader")
def transform_tf_dataloader(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Rewrite ``DataLoader`` construction into a chained dataset pipeline.

  Resolves the target Dataset class path strictly from ``ctx.current_variant``
  or semantic lookup for ``"DataLoader"`` (e.g., ``tf.data.Dataset``), aborting
  safely if unmapped.

  Args:
      node: Original ``DataLoader`` call node.
      ctx: HookContext providing ``current_variant`` and API lookup.

  Returns:
      Transformed method chain, or ``node`` unchanged if unmapped or empty.
  """
  dataset_api = resolve_variant_api(ctx, "DataLoader")
  if not dataset_api:
    return node

  dataset_expr = _extract_dataset_expr(node.args)
  if dataset_expr is None:
    return node

  batch_size_arg = _get_arg_by_name(node.args, "batch_size")
  shuffle_arg = _get_arg_by_name(node.args, "shuffle")

  tensors = _extract_tensor_dataset_inputs(dataset_expr) or [dataset_expr]
  slice_input: cst.BaseExpression = (
    cst.Tuple(elements=[cst.Element(t) for t in tensors]) if len(tensors) > 1 else tensors[0]
  )

  from_slices = create_dotted_name(f"{dataset_api}.from_tensor_slices")
  autotune_path = f"{dataset_api.rsplit('.', 1)[0]}.AUTOTUNE" if "." in dataset_api else "AUTOTUNE"

  current_chain = cst.Call(func=from_slices, args=[cst.Arg(slice_input)])

  if shuffle_arg and isinstance(shuffle_arg.value, cst.Name) and shuffle_arg.value.value == "True":
    current_chain = cst.Call(
      func=cst.Attribute(value=current_chain, attr=cst.Name("shuffle")),
      args=[
        cst.Arg(
          keyword=cst.Name("buffer_size"),
          value=cst.Integer("1024"),
          equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
          ),
        )
      ],
    )

  batch_val = batch_size_arg.value if batch_size_arg else cst.Integer("1")
  current_chain = cst.Call(
    func=cst.Attribute(value=current_chain, attr=cst.Name("batch")),
    args=[cst.Arg(value=batch_val)],
  )

  return cst.Call(
    func=cst.Attribute(value=current_chain, attr=cst.Name("prefetch")),
    args=[cst.Arg(value=create_dotted_name(autotune_path))],
  )
