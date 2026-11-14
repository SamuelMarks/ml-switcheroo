"""Functional Array-API Fallback Decomposition Plugin for ml-switcheroo.

Routes operations lacking a direct target mapping through the registered
functional Array-API IR namespace discovered dynamically from ``FrameworkAdapter``
traits and ``HookContext.current_variant`` (with zero hardcoded framework paths).
"""

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import ImportConfig, available_frameworks, get_adapter
from ml_switcheroo.plugins.utils import create_dotted_name, is_valid_dotted_api, resolve_variant_api


def _discover_array_ir_namespace(ctx: HookContext) -> str | None:
  """Discover the canonical Array-API decomposition namespace from registered adapters.

  Args:
      ctx: Active ``HookContext``.

  Returns:
      Dotted Array-API module path (e.g., from the registered XLA/Array adapter), or ``None``.
  """
  configured = ctx.raw_config("decompose_namespace", default=None)
  if isinstance(configured, str) and is_valid_dotted_api(configured):
    return configured

  for fw_name in sorted(available_frameworks()):
    adapter = get_adapter(fw_name)
    if adapter is None:
      continue
    try:
      traits = adapter.plugin_traits
      inherits_from = adapter.inherits_from
      namespaces = adapter.import_namespaces
      alias_mod = adapter.import_alias[0]
    except AttributeError:
      continue
    if traits.has_numpy_compatible_arrays and traits.requires_explicit_rng and inherits_from is None:
      for ns_path, ns_cfg in namespaces.items():
        is_array_tier = (
          ns_cfg.tier == SemanticTier.ARRAY_API
          if isinstance(ns_cfg, ImportConfig)
          else ns_cfg.get("tier") == SemanticTier.ARRAY_API.value
        )
        if is_array_tier and is_valid_dotted_api(ns_path):
          return ns_path
      if is_valid_dotted_api(alias_mod):
        return alias_mod
  return None


@register_hook("jax_decompose")
def decompose_via_jax(node: cst.Call, ctx: HookContext) -> cst.Call | cst.BaseExpression:
  """Rewrite an unmapped API call through ``current_variant`` or Array-API IR semantics.

  Args:
      node: Original CST Call node lacking direct target support.
      ctx: Hook context containing active operation and framework metadata.

  Returns:
      Rewritten ``cst.Call`` node, or ``node`` unchanged if no IR namespace is registered.
  """
  op_name = ctx.current_op_id if isinstance(ctx.current_op_id, str) and ctx.current_op_id else "UnknownOp"
  target_api = resolve_variant_api(ctx) if ctx.current_op_id else None

  if not target_api:
    ir_ns = _discover_array_ir_namespace(ctx)
    if not ir_ns:
      return node
    target_api = f"{ir_ns}.{op_name.lower()}"

  return node.with_changes(func=create_dotted_name(target_api))
