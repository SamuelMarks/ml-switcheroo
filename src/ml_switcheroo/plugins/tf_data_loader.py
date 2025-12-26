"""
Plugin for Native TensorFlow Data Pipeline Generation.

Converts generic `DataLoader` usage into `tf.data.Dataset` pipelines.
Blind execution based on wiring.
"""

import libcst as cst
from typing import List, Optional, Union
from ml_switcheroo.core.hooks import register_hook, HookContext


def _get_arg_by_name(args: List[cst.Arg], name: str) -> Optional[cst.Arg]:
  for arg in args:
    if arg.keyword and arg.keyword.value == name:
      return arg
  return None


def _extract_tensor_dataset_inputs(node: cst.BaseExpression) -> Optional[List[cst.BaseExpression]]:
  """
  Heuristic to unwrap TensorDataset(x, y) into [x, y].
  Used to pass tensors directly to tf.data.Dataset.from_tensor_slices.
  """
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
  """
  Plugin Hook: Rewrites DataLoader construction into tf.data chain.

  Triggers if mapped via `requires_plugin`.
  Input: `DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)`
  Output: `tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(32).prefetch(AUTOTUNE)`
  """
  if not node.args:
    return node

  # 1. Parse Arguments
  dataset_arg = node.args[0].value
  batch_size_arg = _get_arg_by_name(node.args, "batch_size")
  shuffle_arg = _get_arg_by_name(node.args, "shuffle")

  # 2. Unwrap Dataset
  tensors = _extract_tensor_dataset_inputs(dataset_arg)
  if not tensors:
    tensors = [dataset_arg]

  # 3. Construct Input Tuple
  if len(tensors) > 1:
    slice_input = cst.Tuple(elements=[cst.Element(t) for t in tensors])
  else:
    slice_input = tensors[0]

  # 4. Build Pipeline Chain
  # Root: tf.data.Dataset.from_tensor_slices
  tf_data_ds = cst.Attribute(value=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("data")), attr=cst.Name("Dataset"))
  from_slices = cst.Attribute(value=tf_data_ds, attr=cst.Name("from_tensor_slices"))

  base = cst.Call(func=from_slices, args=[cst.Arg(slice_input)])
  current_chain = base

  # .shuffle(buffer_size=1024)
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

  # .batch(batch_size)
  batch_val = batch_size_arg.value if batch_size_arg else cst.Integer("1")
  current_chain = cst.Call(
    func=cst.Attribute(value=current_chain, attr=cst.Name("batch")), args=[cst.Arg(value=batch_val)]
  )

  # .prefetch(AUTOTUNE)
  current_chain = cst.Call(
    func=cst.Attribute(value=current_chain, attr=cst.Name("prefetch")),
    args=[
      cst.Arg(
        value=cst.Attribute(value=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("data")), attr=cst.Name("AUTOTUNE"))
      )
    ],
  )

  return current_chain
