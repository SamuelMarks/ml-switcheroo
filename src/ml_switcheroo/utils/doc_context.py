"""
Documentation Context Builder.

This module provides the logic to transform internal semantic definitions
(from the ``SemanticsManager``) into a structured View Model suitable for
rendering documentation pages (ReStructuredText/HTML).

It handles:
- formatting argument lists.
- resolving implementation strategies (Direct vs Plugin vs Macro).
- resolving documentation URLs via Framework Adapters.
"""

from typing import Any, Dict, List, Optional, Union

from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import get_framework_priority_order


class DocContextBuilder:
  """
  Prepares view data for operation documentation pages.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initialize the builder.

    Args:
        semantics: Valid SemanticsManager instance.
    """
    self.semantics = semantics

  def build(self, op_name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs the documentation context for a single operation.

    Args:
        op_name: The abstract operation ID (e.g. "Linear").
        definition: The raw dictionary definition from the SemanticsManager.

    Returns:
        Dict: A structural dictionary containing:
            - name (str)
            - description (str)
            - args (List[str]): Formatted signature strings.
            - variants (List[Dict]): Implementation details per framework.
    """
    args_list = self._format_args(definition.get("std_args", []))
    variants = self._resolve_variants(definition.get("variants", {}))

    return {
      "name": op_name,
      "description": definition.get("description", "No description available.").strip(),
      "args": args_list,
      "variants": variants,
    }

  def _format_args(self, std_args: List[Any]) -> List[str]:
    """
    Formats the standard arguments list into human-readable signature strings.

    Handles various ODL formats:
    - String: "x"
    - Tuple: ("x", "int")
    - Dict: {"name": "x", "type": "int", "default": "-1"}

    Returns:
        List of strings like ["x: Tensor", "dim: int = -1"].
    """
    formatted = []
    for arg in std_args:
      name = "unknown"
      type_hint = None
      default_val = None

      if isinstance(arg, str):
        name = arg
      elif isinstance(arg, (list, tuple)) and len(arg) > 0:
        name = arg[0]
        if len(arg) > 1:
          type_hint = arg[1]
      elif isinstance(arg, dict):
        name = arg.get("name", "unknown")
        type_hint = arg.get("type")
        default_val = arg.get("default")

      # Construct string
      sig_part = name
      if type_hint and type_hint != "Any":
        sig_part += f": {type_hint}"
      if default_val is not None:
        sig_part += f" = {default_val}"

      formatted.append(sig_part)

    return formatted

  def _resolve_variants(self, variants_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes the raw variants dictionary into display-ready objects.

    Sorts frameworks by UI priority, retrieves display names,
    classifies implementation types, and resolves doc URLs.

    Args:
        variants_map: Dictionary of {framework_key: variant_def}.
                      Note: variant_def can be None (explicitly unsupported).

    Returns:
        List of dicts representing supported variants.
    """
    results = []
    priority_order = get_framework_priority_order()

    # Filter keys that are actually in variants map, sorted by global priority
    sorted_keys = [k for k in priority_order if k in variants_map]
    # Add any explicit keys not in priority list at the end
    unsorted_keys = [k for k in variants_map if k not in priority_order]
    final_order = sorted_keys + sorted(unsorted_keys)

    for fw_key in final_order:
      variant_def = variants_map[fw_key]
      # Skip explicitly null variants (unsupported)
      if not variant_def:
        continue

      adapter = get_adapter(fw_key)
      display_name = getattr(adapter, "display_name", fw_key) if adapter else fw_key

      api_path = variant_def.get("api", "—")
      impl_type = self._determine_impl_type(variant_def)

      doc_url = None
      if adapter and api_path != "—" and "Plugin" not in impl_type and "Macro" not in impl_type:
        doc_url = adapter.get_doc_url(api_path)

      results.append(
        {
          "framework": display_name,
          "key": fw_key,
          "api": api_path,
          "doc_url": doc_url,
          "implementation_type": impl_type,
          "raw_def": variant_def,  # Keep raw for advanced template logic if needed
        }
      )

    return results

  def _determine_impl_type(self, variant: Dict[str, Any]) -> str:
    """
    Classifies the implementation strategy.

    Args:
        variant: The variant definition dictionary.

    Returns:
        A readable string string (e.g. "Direct Mapping", "Plugin (...)").
    """
    if "requires_plugin" in variant:
      return f"Plugin ({variant['requires_plugin']})"

    if "macro_template" in variant:
      return f"Macro '{variant['macro_template']}'"

    if variant.get("transformation_type") == "infix":
      op = variant.get("operator", "?")
      return f"Infix ({op})"

    if variant.get("transformation_type") == "inline_lambda":
      return "Inline Lambda"

    if variant.get("api"):
      return "Direct Mapping"

    return "Custom / Partial"
