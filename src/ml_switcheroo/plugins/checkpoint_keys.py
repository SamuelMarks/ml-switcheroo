"""
Plugin for Checkpoint Key Remapping.

Handles the runtime impedance mismatch between PyTorch state dictionaries and
Flax/JAX parameter trees.

Problem:
1. Naming: Torch uses `layer.0.weight`. Flax uses `layer_0.kernel` (or `scale` for BN).
2. Shapes: Torch Linear/Conv weights are usually `(Out, In, ...)` or `(Out, In)`.
   Flax weights are `(..., In, Out)`.
3. Semantics: `model.load_state_dict(d)` mutates state in Torch. JAX requires returning a new tree.

Solution:
This plugin converts `model.load_state_dict(state)` calls into usage of a generated runtime
utility `KeyMapper.from_torch(state)`.

It creates the dependency on the `KeyMapper` class, which the transpiler's PreambleGenerator
is expected to inject into the output file using the source code defined in `KEY_MAPPER_SOURCE`.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext

# The runtime utility source code to be injected by the transpiler
KEY_MAPPER_SOURCE = r"""
import jax.numpy as jnp
import numpy as np
import re

class KeyMapper:
    @staticmethod
    def map_name(name):
        # 1. Standardize replacements
        name = name.replace("weight", "kernel")
        name = name.replace("running_mean", "mean")
        name = name.replace("running_var", "var")
        # BN scale convention
        if "bn" in name or "norm" in name:
            name = name.replace("kernel", "scale")
        
        # 2. Separators: layer.0 -> layer_0
        parts = name.split(".")
        new_parts = []
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0:
                # Merge with previous: layer.0 -> layer_0
                prev = new_parts.pop()
                new_parts.append(f"{prev}_{p}")
            else:
                new_parts.append(p)
        return ".".join(new_parts)

    @staticmethod
    def map_value(key, val):
        val = np.array(val)
        # Heuristic Transpose for Dense/Conv kernels
        # Torch Linear: (Out, In) -> JAX (In, Out)
        # Torch Conv2d: (Out, In, H, W) -> JAX (H, W, In, Out)
        
        if "weight" in key or "kernel" in key:
            if val.ndim == 2:
                # Linear
                val = val.transpose((1, 0))
            elif val.ndim == 4:
                # Conv2d: (O, I, H, W) -> (H, W, I, O)
                val = val.transpose((2, 3, 1, 0))
        return jnp.array(val)

    @classmethod
    def from_torch(cls, state_dict):
        # Flattened Torch dict -> Nested Flax dict (simplified)
        # In reality, Flax usually requires unfreezing a target tree and mapping,
        # but here we produce a flat dict with mapped keys/values for loading logic.
        new_dict = {}
        for k, v in state_dict.items():
            nk = cls.map_name(k)
            nv = cls.map_value(k, v)
            new_dict[nk] = nv
        return new_dict
"""


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("checkpoint_mapper")
def transform_checkpoint_keys(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Transforms load_state_dict calls to KeyMapper usage.

  Transformation:
      Input:  `model.load_state_dict(state, strict=True)`
      Output: `variables = KeyMapper.from_torch(state)`

  Note: In JAX/Flax, one does not simply 'load' into a model in-place.
  One gets a variable dict. This transformation implies the user variable `model`
  might need to be reassigned or the result assigned to a parameter variable.

  Since we cannot infer the exact variable to assign to in a generic expression transform,
  we act on the expression itself. If used as `model.load_state_dict(...)`,
  it becomes `KeyMapper.from_torch(...)`.
  """
  if ctx.target_fw not in ["jax", "flax", "flax_nnx"]:
    return node

  # Identify 'state_dict' argument (usually arg 0)
  args = list(node.args)
  if not args:
    return node

  state_arg = args[0]  # Assume pos arg 0 is state_dict

  # Check for keyword 'state_dict'
  for arg in args:
    if arg.keyword and arg.keyword.value == "state_dict":
      state_arg = arg
      break

  # Clean arg (remove keyword if present in transformation target)
  # Removing forcing of comma to avoid (sd, ) syntax in output
  clean_state_arg = state_arg.with_changes(keyword=None, equal=cst.MaybeSentinel.DEFAULT, comma=cst.MaybeSentinel.DEFAULT)

  # Build: KeyMapper.from_torch(state_arg)
  mapper_func = cst.Attribute(value=cst.Name("KeyMapper"), attr=cst.Name("from_torch"))

  return cst.Call(func=mapper_func, args=[clean_state_arg])
