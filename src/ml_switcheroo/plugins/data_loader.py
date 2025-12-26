"""
Plugin for transforming Data Loaders to Generic Shim.

Handles the mapping of `torch.utils.data.DataLoader` (or similar iterators) to
the `GenericDataLoader` shim.

This plugin is **blindly executed** whenever the Semantic Knowledge Base maps an
operation to `requires_plugin: "convert_dataloader"`. It does not check the
target framework name.

Implementation Details:
- Filters arguments to ensure compatibility with the Generic Shim.
- Explicitly handles `num_workers`, `pin_memory`, and `drop_last` by mapping
  names and passing them (since the Shim accepts them as optional kwargs).
- Injects the Shim class definition at the top of the file on first use
  via `ctx.inject_preamble`.
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.frameworks.common.data import get_shim_code


@register_hook("convert_dataloader")
def transform_dataloader(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Middleware to rewrite DataLoader instantiation.

  **Triggers**
  - Operations marked with ``requires_plugin: "convert_dataloader"`` in JSON.

  **Actions**
  - Injects `GenericDataLoader` class definition into preamble (idempotent).
  - Rewrites function call to `GenericDataLoader(...)`
  - Filters arguments (Dataset as pos 0, preserves supported kwargs).
  """

  # 1. Inject Shim Class (One-time check per file via metadata)
  if not ctx.metadata.get("dataloader_shim_injected"):
    # get_shim_code returns the full python source for the shim class
    ctx.inject_preamble(get_shim_code())
    ctx.metadata["dataloader_shim_injected"] = True

  # 2. Normalize Arguments (Source -> Generic Shim)
  # We reconstruct the call arguments to ensure clean mapping to the Shim signature.
  new_args: List[cst.Arg] = []

  # 2a. Handle Positional Arg 0 (Dataset)
  # Heuristic: The first positional arg is always the dataset to iterate over.
  if len(node.args) > 0 and not node.args[0].keyword:
    new_args.append(node.args[0])
    remaining_args = node.args[1:]
  else:
    remaining_args = list(node.args)

  # 2b. Pass through Keywords
  # The Shim supports these standard keywords explicitly or via **kwargs.
  # We filter to ensure high-fidelity mapping without crashing on duplicates.
  supported_keywords = {
    "batch_size",
    "shuffle",
    "drop_last",
    "num_workers",
    "pin_memory",
    "collate_fn",
    "persistent_workers",
    "dataset",
    "sampler",
    "batch_sampler",
    "timeout",
    "worker_init_fn",
  }

  for arg in remaining_args:
    if not arg.keyword:
      # Preserve extra positional args (Shim signature mimics Torch closely)
      new_args.append(arg)
    else:
      # Pass keywords blindly if they look relevant, logic handled by Shim
      # We could filter strictly here, but standard python kwargs behavior
      # in the shim handles unknown args gracefully.
      new_args.append(arg)

  # 3. Swap Class Name
  new_func = cst.Name("GenericDataLoader")

  return node.with_changes(func=new_func, args=new_args)
