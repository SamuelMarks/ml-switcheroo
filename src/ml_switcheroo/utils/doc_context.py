"""Documentation Context Builder.

This module provides the logic to transform internal semantic definitions
(from the ``SemanticsManager``) into a structured View Model suitable for
rendering documentation pages (ReStructuredText/HTML).

It handles:
- formatting argument lists.
- resolving implementation strategies (Direct vs Plugin vs Macro).
- resolving documentation URLs via Framework Adapters.
"""

from ml_switcheroo.enums import RuntimeTensorOrValue
import typing

from pydantic import JsonValue

from ml_switcheroo.config import get_framework_priority_order
from ml_switcheroo.frameworks import get_adapter

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class DocContextBuilder:
  """Prepare view data for operation documentation pages.

  This builder transforms internal operation definitions into a structured
  view context dictionary that can be used directly by documentation generators.
  """

  def __init__(self, semantics: "SemanticsManager") -> None:
    """Initialize the builder.

    Args:
        semantics: Valid SemanticsManager instance.

    """
    self.semantics = semantics

  def build(self, op_name: str, definition: typing.Mapping[str, RuntimeTensorOrValue]) -> dict[str, JsonValue]:
    """Construct the documentation context for a single operation.

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
    raw_args = definition.get("std_args", [])
    args_seq = raw_args if isinstance(raw_args, list) else []
    args_list: list[JsonValue] = list(self._format_args(args_seq))
    raw_vars = definition.get("variants", {})
    vars_map = raw_vars if isinstance(raw_vars, dict) else {}
    variants: list[JsonValue] = list(self._resolve_variants(vars_map))
    raw_desc = definition.get("description", "No description available.")
    desc_str = str(raw_desc).strip() if isinstance(raw_desc, str) else "No description available."

    return {
      "name": op_name,
      "description": desc_str,
      "args": args_list,
      "variants": variants,
    }

  def _format_args(self, std_args: typing.Sequence[RuntimeTensorOrValue]) -> list[str]:
    """Format the standard arguments list into human-readable signature strings.

    Handles various ODL formats:
    - String: "x"
    - Tuple: ("x", "int")
    - Dict: {"name": "x", "type": "int", "default": "-1"}

    Args:
        std_args: List of standard arguments in various ODL formats.

    Returns:
        List of strings like ["x: Tensor", "dim: int = -1"].

    """
    formatted = []
    for arg in std_args:
      name = "unknown"
      type_hint: str | None = None
      default_val: RuntimeTensorOrValue = None

      if isinstance(arg, str):
        name = arg
      elif isinstance(arg, (list, tuple)) and len(arg) > 0:
        name = str(arg[0])
        if len(arg) > 1 and arg[1] is not None:
          type_hint = str(arg[1])
      elif isinstance(arg, dict):
        name = str(arg.get("name", "unknown"))
        raw_t = arg.get("type")
        type_hint = str(raw_t) if raw_t is not None else None
        default_val = arg.get("default")

      if name.startswith("__"):
        continue

      # Construct string
      sig_part = name
      if type_hint and type_hint != "Any":
        sig_part += f": {type_hint}"
      if default_val is not None:
        clean_val = str(default_val).replace("`", "")
        sig_part += f" = {clean_val}"

      formatted.append(sig_part)

    return formatted

  def _resolve_variants(self, variants_map: typing.Mapping[str, RuntimeTensorOrValue]) -> list[dict[str, JsonValue]]:
    """Process the raw variants dictionary into display-ready objects.

    Sorts frameworks by UI priority, retrieves display names,
    classifies implementation types, and resolves doc URLs.

    Args:
        variants_map: Dictionary of {framework_key: variant_def}.
                      Note: variant_def can be None (explicitly unsupported).

    Returns:
        List of dicts representing supported variants.

    """
    results: list[dict[str, JsonValue]] = []
    priority_order = get_framework_priority_order()

    # Filter keys that are actually in variants map, sorted by global priority
    sorted_keys = [k for k in priority_order if k in variants_map]
    # Add any explicit keys not in priority list at the end
    unsorted_keys = [k for k in variants_map if k not in priority_order]
    final_order = sorted_keys + sorted(unsorted_keys)

    for fw_key in final_order:
      variant_def = variants_map[fw_key]
      # Skip explicitly null variants (unsupported)
      if not isinstance(variant_def, dict) or not variant_def:
        continue

      adapter = get_adapter(fw_key)
      display_name = getattr(adapter, "display_name", fw_key) if adapter else fw_key

      raw_api = variant_def.get("api", "—")
      api_path = str(raw_api) if raw_api is not None else "—"
      impl_type = self._determine_impl_type(variant_def)

      doc_url: str | None = None
      if adapter and api_path != "—" and "Plugin" not in impl_type and "Macro" not in impl_type:
        doc_url = adapter.get_doc_url(api_path)

      results.append(
        {
          "framework": str(display_name),
          "key": fw_key,
          "api": api_path,
          "doc_url": doc_url,
          "implementation_type": impl_type,
          "raw_def": dict(variant_def),  # Keep raw for advanced template logic if needed
        }
      )

    return results

  def _determine_impl_type(self, variant: typing.Mapping[str, RuntimeTensorOrValue]) -> str:
    """Classify the implementation strategy.

    Args:
        variant: The variant definition dictionary.

    Returns:
        A readable string (e.g. "Direct Mapping", "Plugin (...)").

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
