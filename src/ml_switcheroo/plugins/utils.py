"""
Shared Utilities for Plugins.

This module provides common helper functions for AST manipulation and
framework detection, decoupling individual plugins from hardcoded lists of libraries.
It relies on the `HookContext` to dynamically resolve framework identities.
"""

import libcst as cst
from typing import Optional

from ml_switcheroo.core.hooks import HookContext


def create_dotted_name(name_str: str) -> cst.BaseExpression:
  """
  Creates a CST Attribute chain from a dotted string.

  Example: "jax.numpy.add" -> Attribute(value=Attribute(value=Name("jax"), ...), ...)

  Args:
      name_str (str): The dot-separated API path.

  Returns:
      cst.BaseExpression: The constructed AST node.
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def is_framework_module_node(node: cst.CSTNode, ctx: HookContext) -> bool:
  """
  Determines if a CST node represents a known framework namespace root.

  This is used to distinguish function calls (e.g. `torch.add(x)`) from method calls
  (e.g. `x.add()`). If the receiver `x` is a variable, this returns False.
  If the receiver is `torch`, it returns True.

  Logic:
      1. Checks the configured Source and Target frameworks in `ctx`.
      2. Checks the Global Semantics Registry (loaded adapters) via `ctx.semantics`.

  Decoupling Note:
      This logic no longer checks for hardcoded strings like "torch" or "jax".
      It relies entirely on the configuration passed in `ctx`.

  Args:
      node (cst.CSTNode): The node to inspect (e.g. the value of an Attribute).
      ctx (HookContext): The execution context containing framework configuration.

  Returns:
      bool: True if the node is a known framework identifier.
  """
  # Extract root name string
  name = _extract_root_name(node)
  if not name:
    return False

  # 1. Check Configured Frameworks (Fast Path)
  # The user has explicitly selected Source and Target frameworks in the config.
  # The roots (e.g. "torch", "jax") are definitely framework modules.
  if ctx.source_fw and name == ctx.source_fw:
    return True
  if ctx.target_fw and name == ctx.target_fw:
    return True

  # 2. Check Semantics Registry (Dynamic)
  # This catches frameworks that are registered but not currently selected as source/target,
  # or secondary roots (e.g. "numpy" when targeting Keras).
  if ctx.semantics:
    # Check loaded framework configs
    configs = getattr(ctx.semantics, "framework_configs", {})

    # Direct match against framework ID (e.g. 'torch', 'tensorflow')
    if name in configs:
      return True

    # Check match against known aliases (e.g. 'tf', 'jnp', 'nn')
    for _fw_key, conf in configs.items():
      # Support both dict and Pydantic object access for robustness
      if isinstance(conf, dict):
        alias_info = conf.get("alias")
      else:
        # Assuming Pydantic model with .alias attribute
        alias_info = getattr(conf, "alias", None)
        if alias_info and hasattr(alias_info, "model_dump"):
          alias_info = alias_info.model_dump()

      if alias_info and isinstance(alias_info, dict):
        if alias_info.get("name") == name:
          return True

    # Check import map roots (e.g. "torch.nn" implies "torch" is known)
    # Replaced legacy import_data usages with _source_registry check
    source_registry = getattr(ctx.semantics, "_source_registry", {})
    for mod_path in source_registry.keys():
      known_root = mod_path.split(".")[0]
      if name == known_root:
        return True

  return False


def _extract_root_name(node: cst.CSTNode) -> Optional[str]:
  """Recursively extracts the root identifier from a Name or Attribute chain."""
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    return _extract_root_name(node.value)
  return None
