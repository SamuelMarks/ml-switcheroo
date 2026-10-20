"""Reverse index building logic for the Semantic Knowledge Base."""

import json
import os
import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class IndexBuilder:
  """Modular reverse-index builder mapping concrete framework APIs to abstract operations."""

  @staticmethod
  def build_index(manager: "SemanticsManager") -> None:
    """Construct the reverse index mapping from concrete API endpoints back to their abstract definitions.

    Args:
        manager: The SemanticsManager instance whose `_reverse_index` and `_variant_cache` will be populated.
    """
    if not hasattr(manager, "_reverse_index"):
      manager._reverse_index = {}
    else:
      manager._reverse_index.clear()

    if not hasattr(manager, "_variant_cache"):
      manager._variant_cache = {}
    else:
      manager._variant_cache.clear()
    alias_map: typing.Dict[str, str] = {}

    import importlib.resources

    try:
      aliases_json_path = importlib.resources.files("ml_ecosystem_snapshots.snapshots").joinpath("aliases.json")
      if aliases_json_path.is_file():
        with aliases_json_path.open("r", encoding="utf-8") as f:
          alias_map.update(json.load(f))
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ModuleNotFoundError):
      pass

    for fw, config in manager.framework_configs.items():
      alias_conf = config.get("alias")
      if isinstance(alias_conf, dict):
        mod = alias_conf.get("module")
        name = alias_conf.get("name")
        if isinstance(mod, str) and mod and isinstance(name, str) and name:
          alias_map[name] = mod

    priority_scores: typing.Dict[str, int] = {}
    priority_json_path = os.path.join(os.path.dirname(__file__), "priority_scores.json")
    if os.path.exists(priority_json_path):
      with open(priority_json_path, "r", encoding="utf-8") as f:
        priority_scores = json.load(f)

    def get_priority(
      abs_id: str,
      details: typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]],
      tier: typing.Optional[str],
    ) -> int:
      """Calculate the disambiguation priority score for an abstract operation definition.

      Args:
          abs_id: Abstract operation identifier string.
          details: Operation definition dictionary containing variants.
          tier: Semantic tier value string ('array', 'neural', 'extras').

      Returns:
          Integer priority score (higher wins collisions in `_reverse_index`).
      """
      score = priority_scores.get(abs_id, 0)
      if tier == SemanticTier.ARRAY_API.value:
        score += 50
      elif tier == SemanticTier.NEURAL.value:
        score -= 50
      elif tier == SemanticTier.EXTRAS.value:
        score -= 100
      raw_vars = details.get("variants") or {}
      score += len(getattr(raw_vars, "keys")())
      return int(score)

    for abstract_id, details in manager.data.items():
      variants = details.get("variants", {})
      tier = manager._key_origins.get(abstract_id)
      score = get_priority(abstract_id, details, tier)

      for _engine, impl in getattr(variants, "items")():
        if not isinstance(impl, dict):
          continue
        api_name = impl.get("api")
        if isinstance(api_name, str) and api_name:

          def register_api(name: str) -> None:
            """Register a concrete API path in `manager._reverse_index` if it has higher priority.

            Args:
                name: Concrete dotted API path string.
            """
            if name in manager._reverse_index:
              existing_id, existing_details = manager._reverse_index[name]
              existing_tier = manager._key_origins.get(existing_id)
              existing_score = get_priority(existing_id, existing_details, existing_tier)
              if score > existing_score:
                manager._reverse_index[name] = (abstract_id, details)
            else:
              manager._reverse_index[name] = (abstract_id, details)

          register_api(api_name)

          parts = api_name.split(".")
          if parts[0] in alias_map:
            fqn = alias_map[parts[0]] + "." + ".".join(parts[1:])
            register_api(fqn)

          for alias_name, module_path in alias_map.items():
            if api_name.startswith(module_path + "."):
              fqn = alias_name + api_name[len(module_path) :]
              register_api(fqn)
