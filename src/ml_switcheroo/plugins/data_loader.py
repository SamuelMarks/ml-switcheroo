"""
Plugin for transforming Data Loaders.

Handles the mapping of `torch.utils.data.DataLoader` to:
1. `GenericDataLoader` shim (for JAX/NumPy).
2. Native `torch.utils.data.DataLoader` (Pass-through for Torch).
3. `tf.data.Dataset` (Future implementation for TF).

It leverages `frameworks.common.data` to fetch the shim implementation when needed.
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.frameworks.common.data import get_shim_code


def _get_arg_val(args: List[cst.Arg], name: str) -> Optional[cst.BaseExpression]:
  """Extract value of a specific keyword arg."""
  for arg in args:
    if arg.keyword and arg.keyword.value == name:
      return arg.value
  return None


@register_hook("convert_dataloader")
def transform_dataloader(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Middleware to rewrite DataLoader instantiation.

  Triggers:
      Operations marked with `requires_plugin: "convert_dataloader"`
      (e.g., `DataLoader` in `k_framework_extras.json`).

  Strategy:
      - If Target == Torch: Keep as is.
      - If Target != Torch: Inject Shim logic and rewrite to `GenericDataLoader`.
  """

  # 1. Passthrough for Torch (Native support)
  if ctx.target_fw == "torch":
    return node

  # 2. Inject Shim for others (JAX, NumPy, etc.)
  # We use a unique key in metadata to ensure we only inject the class definition once per file.
  if not ctx.metadata.get("dataloader_shim_injected"):
    ctx.inject_preamble(get_shim_code())
    ctx.metadata["dataloader_shim_injected"] = True

  # 3. Normalize Arguments (Source -> Abstract)
  # We extract arguments from the source call based on PyTorch conventions
  # and map them to the GenericDataLoader signature.

  # Defaults
  dataset_arg = None
  batch_size_arg = None
  shuffle_arg = None
  drop_last_arg = None

  # Parse Source Args (Heuristic: PyTorch signature is (dataset, batch_size=1, shuffle=False...))
  # Positional mapping
  if len(node.args) > 0 and not node.args[0].keyword:
    dataset_arg = node.args[0].value

  if len(node.args) > 1 and not node.args[1].keyword:
    batch_size_arg = node.args[1].value

  # Keyword mapping
  dataset_arg = _get_arg_val(node.args, "dataset") or dataset_arg
  batch_size_arg = _get_arg_val(node.args, "batch_size") or batch_size_arg
  shuffle_arg = _get_arg_val(node.args, "shuffle")
  drop_last_arg = _get_arg_val(node.args, "drop_last")

  # 4. Construct New Call
  new_args = []

  # Dataset is mandatory
  if dataset_arg:
    new_args.append(cst.Arg(value=dataset_arg))
  else:
    # If we couldn't find dataset, maybe the user passed it as kwargs or positional logic failed.
    # Fallback: pass all original args and hope for the best?
    # Safer: Pass original args but rename function.
    # Most users use (dataset, ...)
    pass

  if batch_size_arg:
    new_args.append(cst.Arg(keyword=cst.Name("batch_size"), value=batch_size_arg))

  if shuffle_arg:
    new_args.append(cst.Arg(keyword=cst.Name("shuffle"), value=shuffle_arg))

  if drop_last_arg:
    new_args.append(cst.Arg(keyword=cst.Name("drop_last"), value=drop_last_arg))

  # If parsing failed completely (no explicit args found but original had args),
  # just pass original args to the shim.
  if not new_args and node.args:
    new_args = node.args

  # 5. Swap Class Name
  new_func = cst.Name("GenericDataLoader")

  return node.with_changes(func=new_func, args=new_args)
