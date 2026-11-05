"""Shared Utilities for Plugins.

This module provides common helper functions for AST manipulation and
framework detection, decoupling individual plugins from hardcoded lists of libraries.
It relies on the `HookContext` to dynamically resolve framework identities.
"""

import typing

import libcst as cst
from pydantic import JsonValue


@typing.runtime_checkable
class _ModelDumpProtocol(typing.Protocol):
  """Protocol for objects providing `model_dump()`."""

  def model_dump(self) -> dict[str, JsonValue]:
    """Dump object fields to dictionary.

    Returns:
        Dictionary of field values.
    """
    ...


@typing.runtime_checkable
class _AliasContainerProtocol(typing.Protocol):
  """Protocol for configuration objects exposing `alias`."""

  alias: _ModelDumpProtocol | dict[str, JsonValue] | None


if typing.TYPE_CHECKING:
  from ml_switcheroo.core.hooks import HookContext


def create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Create a CST Attribute chain from a dotted string.

  Example: "jax.numpy.add" -> Attribute(value=Attribute(value=Name("jax"), ...), ...)

  Args:
      name_str (str): The dot-separated API path.

  Returns:
      cst.BaseExpression: The constructed AST node.

  """
  parts = name_str.split(".")
  node: cst.Name | cst.Attribute = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def is_framework_module_node(node: cst.CSTNode, ctx: "HookContext") -> bool:
  """Determine if a CST node represents a known framework namespace root.

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
  if ctx.source_framework and name == ctx.source_framework:
    return True
  if ctx.target_framework and name == ctx.target_framework:
    return True

  # 2. Check Semantics Registry (Dynamic)
  # This catches frameworks that are registered but not currently selected as source/target,
  # or secondary roots (e.g. "numpy" when targeting Keras).
  if ctx.semantics is not None:
    # Check loaded framework configs
    raw_configs = ctx.semantics.framework_configs
    configs = raw_configs or {}

    # Direct match against framework ID (e.g. 'torch', 'tensorflow')
    if name in configs:
      return True

    # Check match against known aliases (e.g. 'tf', 'jnp', 'nn')
    for _fw_key, conf_obj in configs.items():
      conf: dict[str, JsonValue] | _AliasContainerProtocol | JsonValue = conf_obj
      # Support both dict and Pydantic object access for robustness
      if isinstance(conf, dict):
        alias_info = conf.get("alias")
      else:
        alias_raw = conf.alias if isinstance(conf, _AliasContainerProtocol) else None
        alias_info = alias_raw.model_dump() if isinstance(alias_raw, _ModelDumpProtocol) else alias_raw

      if alias_info and isinstance(alias_info, dict):
        if alias_info.get("name") == name:
          return True

    # Check import map roots (e.g. "torch.nn" implies "torch" is known)
    # Replaced legacy import_data usages with _source_registry check
    try:
      source_registry = ctx.semantics._source_registry
    except AttributeError:
      source_registry = {}
    for mod_path in source_registry.keys():
      known_root = mod_path.split(".")[0]
      if name == known_root:
        return True

  return False


def _extract_root_name(node: cst.CSTNode) -> str | None:
  """Recursively extracts the root identifier from a Name or Attribute chain.

  Args:
      node (cst.CSTNode): The CST node (Name or Attribute) to inspect.

  Returns:
      Optional[str]: The name of the root identifier if found; otherwise, None.
  """
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    return _extract_root_name(node.value)
  return None
