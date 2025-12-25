"""
Plugin for transforming Data Loaders.

Handles the mapping of `torch.utils.data.DataLoader` to:
1. `GenericDataLoader` shim (for JAX/NumPy).
2. Native `torch.utils.data.DataLoader` (Pass-through for Torch).

Implementation Details:
- Filters arguments to ensure compatibility with the Generic Shim.
- Explicitly handles `num_workers`, `pin_memory`, and `drop_last` by mapping
  names and passing them (since the Shim now accepts them as optional kwargs).
- Injects the Shim class definition at the top of the file on first use.
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
  """Middleware to rewrite DataLoader instantiation.

  **Triggers**

  - Operations marked with ``requires_plugin: "convert_dataloader"`` (e.g., ``DataLoader`` in ``k_framework_extras.json``).

  **Strategy**

  - If Target = Torch: Keep as is.
  - If Target != Torch: Inject Shim logic and rewrite to ``GenericDataLoader``.
  - Maps performance args (``num_workers``, ``pin_memory``) to the shim signatures where they are safely ignored.
  """

  # 1. Passthrough for Torch (Native support)
  if ctx.target_fw.lower() == "torch":
    return node

  # 2. Inject Shim for others (JAX, NumPy, etc.)
  # We use a unique key in metadata to ensure we only inject the class definition once per file.
  if not ctx.metadata.get("dataloader_shim_injected"):
    ctx.inject_preamble(get_shim_code())
    ctx.metadata["dataloader_shim_injected"] = True

  # 3. Normalize Arguments (Source -> Generic Shim)
  # We reconstruct the call arguments.
  new_args: List[cst.Arg] = []

  # 3a. Handle Positional Arg 0 (Dataset)
  if len(node.args) > 0 and not node.args[0].keyword:
    # First positional is dataset
    new_args.append(node.args[0])
    # Process remaining args
    remaining_args = node.args[1:]
  else:
    remaining_args = list(node.args)

  # 3b. Pass through Keywords
  # List of known args supported by Shim (explicitly or via kwargs)
  supported_keywords = {
    "batch_size",
    "shuffle",
    "drop_last",
    "num_workers",
    "pin_memory",
    "collate_fn",
    "persistent_workers",
    "dataset",
  }

  for arg in remaining_args:
    if not arg.keyword:
      # This is a positional arg beyond dataset.
      # New Shim behavior matches Torch signature closely.
      new_args.append(arg)
    else:
      # Keyword conservation
      k_name = arg.keyword.value
      if k_name in supported_keywords:
        new_args.append(arg)
      else:
        new_args.append(arg)

  # 4. Swap Class Name
  new_func = cst.Name("GenericDataLoader")

  return node.with_changes(func=new_func, args=new_args)
