"""
Plugin for Native TensorFlow Data Pipeline Generation.
"""

import libcst as cst
from typing import List, Optional, Tuple, Union
from ml_switcheroo.core.hooks import register_hook, HookContext


def _get_arg_by_name(args: List[cst.Arg], name: str) -> Optional[cst.Arg]:
  for arg in args:
    if arg.keyword and arg.keyword.value == name:
      return arg
  return None


def _extract_tensor_dataset_inputs(node: cst.BaseExpression) -> Optional[List[cst.BaseExpression]]:
  if isinstance(node, cst.Call):
    func_name = ""
    if isinstance(node.func, cst.Name):
      func_name = node.func.value
    elif isinstance(node.func, cst.Attribute):
      func_name = node.func.attr.value

    if func_name == "TensorDataset":
      return [arg.value for arg in node.args]

  return None


@register_hook("tf_data_loader")
def transform_tf_dataloader(node: cst.Call, ctx: HookContext) -> Union[cst.Call, cst.FlattenSentinel]:
  if ctx.target_fw != "tensorflow":
    return node

  if not node.args:
    return node

  dataset_arg = node.args[0].value
  batch_size_arg = _get_arg_by_name(node.args, "batch_size")
  shuffle_arg = _get_arg_by_name(node.args, "shuffle")

  tensors = _extract_tensor_dataset_inputs(dataset_arg)
  if not tensors:
    tensors = [dataset_arg]

  if len(tensors) > 1:
    slice_input = cst.Tuple(elements=[cst.Element(t) for t in tensors])
  else:
    slice_input = tensors[0]

  # tf.data.Dataset
  tf_data_ds = cst.Attribute(value=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("data")), attr=cst.Name("Dataset"))
  # .from_tensor_slices
  from_slices = cst.Attribute(value=tf_data_ds, attr=cst.Name("from_tensor_slices"))

  base = cst.Call(func=from_slices, args=[cst.Arg(slice_input)])
  current_chain = base

  if shuffle_arg and getattr(shuffle_arg.value, "value", "") == "True":
    current_chain = cst.Call(
      func=cst.Attribute(value=current_chain, attr=cst.Name("shuffle")),
      args=[
        cst.Arg(
          keyword=cst.Name("buffer_size"),
          value=cst.Integer("1024"),
          equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        )
      ],
    )

  if batch_size_arg:
    current_chain = cst.Call(
      func=cst.Attribute(value=current_chain, attr=cst.Name("batch")), args=[cst.Arg(value=batch_size_arg.value)]
    )
  else:
    current_chain = cst.Call(
      func=cst.Attribute(value=current_chain, attr=cst.Name("batch")), args=[cst.Arg(value=cst.Integer("1"))]
    )

  current_chain = cst.Call(
    func=cst.Attribute(value=current_chain, attr=cst.Name("prefetch")),
    args=[
      cst.Arg(
        value=cst.Attribute(value=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("data")), attr=cst.Name("AUTOTUNE"))
      )
    ],
  )

  return current_chain
