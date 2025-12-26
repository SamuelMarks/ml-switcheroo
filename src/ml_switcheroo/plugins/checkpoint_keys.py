"""
Plugin for Checkpoint Key Remapping.

Handles the runtime impedance mismatch between PyTorch state dictionaries and
flattened parameter trees (e.g., JAX/Flax).

Logic:
This plugin blindly converts `load_state_dict(state)` calls into usage of a
generated runtime utility `KeyMapper.from_torch(state)`.

It does **not** check the target framework. If a semantic map points to
`requires_plugin="checkpoint_mapper"`, the transformation is applied.
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
    try: 
        val = np.array(val) 
    except: 
        return val # Fallback

    # Heuristic Transpose for Dense/Conv kernels
    if "weight" in key or "kernel" in key: 
      if val.ndim == 2: 
        # Linear: (Out, In) -> (In, Out) 
        val = val.transpose((1, 0)) 
      elif val.ndim == 4: 
        # Conv2d: (O, I, H, W) -> (H, W, I, O) 
        val = val.transpose((2, 3, 1, 0)) 
    return jnp.array(val) 

  @classmethod
  def from_torch(cls, state_dict): 
      new_dict = {} 
      for k, v in state_dict.items(): 
        nk = cls.map_name(k) 
        nv = cls.map_value(k, v) 
        new_dict[nk] = nv
      return new_dict
"""


@register_hook("checkpoint_mapper")
def transform_checkpoint_keys(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Hook: Transforms load_state_dict calls to KeyMapper usage.

  **Transformation**
  `model.load_state_dict(state, strict=True)` -> `KeyMapper.from_torch(state)`

  Triggers blindly if mapped. Injects KeyMapper source code once.
  """
  # 1. Identify 'state_dict' argument (usually arg 0)
  args = list(node.args)
  if not args:
    return node

  state_arg = args[0]  # Assume pos arg 0 is state_dict

  # Check for keyword 'state_dict'
  for arg in args:
    if arg.keyword and arg.keyword.value == "state_dict":
      state_arg = arg
      break

  # 2. Inject Preamble Logic (if not already present)
  # Uses a unique key in metadata to prevent duplication
  if not ctx.metadata.get("key_mapper_injected"):
    ctx.inject_preamble(KEY_MAPPER_SOURCE)
    ctx.metadata["key_mapper_injected"] = True

  # 3. Clean arg (remove keyword if present in transformation target)
  clean_state_arg = state_arg.with_changes(keyword=None, equal=cst.MaybeSentinel.DEFAULT, comma=cst.MaybeSentinel.DEFAULT)

  # 4. Build: KeyMapper.from_torch(state_arg)
  mapper_func = cst.Attribute(value=cst.Name("KeyMapper"), attr=cst.Name("from_torch"))

  return cst.Call(func=mapper_func, args=[clean_state_arg])
