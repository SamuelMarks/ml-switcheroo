"""Shared Utilities for Plugins.

This module provides common helper functions for AST manipulation, variant API
lookup, and framework detection, decoupling individual plugins from hardcoded
framework paths and source-only call arity assumptions.
"""

import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.dsl import FrameworkVariant


@typing.runtime_checkable
class _ModelDumpProtocol(typing.Protocol):
  """Protocol for objects providing ``model_dump()``."""

  def model_dump(self) -> dict[str, JsonValue]:
    """Dump object fields to dictionary.

    Returns:
        Dictionary of field values.
    """
    ...


@typing.runtime_checkable
class _AliasContainerProtocol(typing.Protocol):
  """Protocol for configuration objects exposing ``alias``.

  Attributes:
      alias: Optional alias configuration mapping or dumpable model.
  """

  alias: _ModelDumpProtocol | dict[str, JsonValue] | None


if typing.TYPE_CHECKING:
  from ml_switcheroo.core.hooks import HookContext


def is_valid_dotted_api(api: str | None) -> typing.TypeGuard[str]:
  """Check whether a candidate string is a valid dotted Python API identifier path.

  Args:
      api: Candidate dotted API string (e.g., ``"jax.numpy.add"``) or ``None``.

  Returns:
      True if ``api`` is a non-empty string of valid Python identifier segments.
  """
  return isinstance(api, str) and bool(api) and all(part.isidentifier() for part in api.split("."))


def resolve_variant_api(
  ctx: "HookContext",
  fallback_op_ids: str | typing.Sequence[str] | None = None,
) -> str | None:
  """Resolve the target API path strictly from ``HookContext`` variant or lookup.

  Queries ``ctx.lookup_api`` for ``ctx.current_op_id`` (and any explicit fallback
  operation IDs) first, and also honors explicit ``FrameworkVariant`` instances
  attached to ``ctx.current_variant`` when ``semantics`` is absent. Never fabricates
  hardcoded framework strings when semantic mappings are missing.

  Args:
      ctx: Active plugin ``HookContext``.
      fallback_op_ids: Optional abstract operation ID or sequence of IDs to query
          when ``ctx.current_op_id`` is unset.

  Returns:
      Validated dotted API string if mapped in semantics, otherwise ``None``.
  """
  try:
    raw_op_id = ctx.current_op_id
  except AttributeError:
    raw_op_id = None

  candidates: list[str] = []
  if isinstance(raw_op_id, str) and raw_op_id:
    candidates.append(raw_op_id)
  if isinstance(fallback_op_ids, str):
    if fallback_op_ids not in candidates:
      candidates.append(fallback_op_ids)
  elif fallback_op_ids is not None:
    for item in fallback_op_ids:
      if item not in candidates:
        candidates.append(item)

  for op_id in candidates:
    looked_up = ctx.lookup_api(op_id)
    if is_valid_dotted_api(looked_up):
      return looked_up

  try:
    sem = ctx.semantics
  except AttributeError:
    sem = None
  if sem is None or not candidates:
    try:
      variant = ctx.current_variant
    except AttributeError:
      variant = None
    if isinstance(variant, FrameworkVariant) and is_valid_dotted_api(variant.api):
      return variant.api

  return None


def resolve_variant_arg_name(ctx: "HookContext", std_name: str, default: str) -> str:
  """Resolve a target keyword argument name from ``HookContext.current_variant.args``.

  Args:
      ctx: Active plugin ``HookContext``.
      std_name: Standard or source parameter name to look up.
      default: Default target parameter name if not overridden in ``current_variant``.

  Returns:
      Target parameter name string.
  """
  try:
    variant = ctx.current_variant
  except AttributeError:
    variant = None
  if isinstance(variant, FrameworkVariant) and variant.args:
    mapped = variant.args.get(std_name)
    if isinstance(mapped, str) and mapped.isidentifier():
      return mapped
  return default


def create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Create a CST Attribute chain from a dotted string.

  Example: ``"jax.numpy.add"`` -> ``Attribute(value=Attribute(value=Name("jax"), ...), ...)``.

  Args:
      name_str: The dot-separated API path.

  Returns:
      The constructed LibCST expression node.
  """
  parts = name_str.split(".")
  node: cst.Name | cst.Attribute = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _matches_semantics_namespace(name: str, ctx: "HookContext") -> bool:
  """Check if ``name`` matches a registered framework key, alias, or source root.

  Args:
      name: Root identifier string extracted from CST node.
      ctx: Active plugin ``HookContext``.

  Returns:
      True if ``name`` is a registered framework module identifier.
  """
  if ctx.semantics is None:
    return False

  configs = ctx.semantics.framework_configs or {}
  if name in configs:
    return True

  for conf_obj in configs.values():
    conf: dict[str, JsonValue] | _AliasContainerProtocol | JsonValue = conf_obj
    if isinstance(conf, dict):
      alias_info = conf.get("alias")
    else:
      alias_raw = conf.alias if isinstance(conf, _AliasContainerProtocol) else None
      alias_info = alias_raw.model_dump() if isinstance(alias_raw, _ModelDumpProtocol) else alias_raw
    if isinstance(alias_info, dict) and alias_info.get("name") == name:
      return True

  try:
    source_registry = ctx.semantics._source_registry
  except AttributeError:
    source_registry = {}
  return any(name == mod_path.split(".")[0] for mod_path in source_registry)


def is_framework_module_node(node: cst.CSTNode, ctx: "HookContext") -> bool:
  """Determine if a CST node represents a known framework namespace root.

  Args:
      node: The node to inspect (e.g., the value of an ``Attribute``).
      ctx: The execution context containing framework configuration.

  Returns:
      True if the node is a known framework identifier.
  """
  name = _extract_root_name(node)
  if not name:
    return False
  if (ctx.source_framework and name == ctx.source_framework) or (ctx.target_framework and name == ctx.target_framework):
    return True
  return _matches_semantics_namespace(name, ctx)


def _extract_root_name(node: cst.CSTNode) -> str | None:
  """Recursively extract the root identifier from a ``Name`` or ``Attribute`` chain.

  Args:
      node: The CST node (``Name`` or ``Attribute``) to inspect.

  Returns:
      The root identifier name if found, otherwise ``None``.
  """
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    return _extract_root_name(node.value)
  return None
