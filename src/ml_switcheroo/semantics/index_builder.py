"""Reverse index building logic for the Semantic Knowledge Base."""

import importlib.resources
import json
import os
import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.enums import RuntimeTensorOrValue

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager

_TIER_SCORE_OFFSETS: dict[str, int] = {
  SemanticTier.ARRAY_API.value: 50,
  SemanticTier.NEURAL.value: -50,
  SemanticTier.EXTRAS.value: -100,
}


class IndexBuilder:
  """Modular reverse-index builder mapping concrete framework APIs to abstract operations."""

  @staticmethod
  def _reset_caches(manager: "SemanticsManager") -> None:
    """Clear or initialize reverse-index and variant caches on ``manager``.

    Args:
        manager: Target ``SemanticsManager`` instance.
    """
    try:
      manager._reverse_index.clear()
    except AttributeError:
      manager._reverse_index = {}

    try:
      manager._variant_cache.clear()
    except AttributeError:
      manager._variant_cache = {}

    try:
      manager._typed_reverse_index.clear()
    except AttributeError:
      manager._typed_reverse_index = {}

    try:
      manager._typed_variant_cache.clear()
    except AttributeError:
      manager._typed_variant_cache = {}

  @staticmethod
  def _load_alias_map(framework_configs: typing.Mapping[str, dict[str, JsonValue]]) -> dict[str, str]:
    """Build alias-to-module mapping from snapshot resources and registered framework configs.

    Args:
        framework_configs: Per-framework configuration dictionaries on ``SemanticsManager``.

    Returns:
        Dictionary mapping short alias prefixes (e.g. ``'jnp'``) to full module paths (e.g. ``'jax.numpy'``).
    """
    alias_map: dict[str, str] = {}
    try:
      aliases_json_path = importlib.resources.files("ml_ecosystem_snapshots.snapshots").joinpath("aliases.json")
      if aliases_json_path.is_file():
        with aliases_json_path.open("r", encoding="utf-8") as f:
          alias_map.update(json.load(f))
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, ModuleNotFoundError):
      pass

    for config in framework_configs.values():
      alias_conf = config.get("alias")
      if isinstance(alias_conf, dict):
        mod = alias_conf.get("module")
        name = alias_conf.get("name")
        if isinstance(mod, str) and mod and isinstance(name, str) and name:
          alias_map[name] = mod
    return alias_map

  @staticmethod
  def _load_priority_scores() -> dict[str, int]:
    """Load base abstract-operation priority overrides from ``priority_scores.json`` if present.

    Returns:
        Dictionary mapping abstract operation IDs to integer score adjustments.
    """
    priority_json_path = os.path.join(os.path.dirname(__file__), "priority_scores.json")
    if os.path.exists(priority_json_path):
      with open(priority_json_path, "r", encoding="utf-8") as f:
        loaded: dict[str, int] = json.load(f)
        return loaded
    return {}

  @staticmethod
  def _get_priority(
    abs_id: str,
    details: typing.Mapping[str, RuntimeTensorOrValue],
    tier: str | None,
    priority_scores: typing.Mapping[str, int],
  ) -> int:
    """Calculate the disambiguation priority score for an abstract operation definition.

    Args:
        abs_id: Abstract operation identifier string.
        details: Operation definition dictionary containing variants.
        tier: Semantic tier value string (``'array'``, ``'neural'``, ``'extras'``).
        priority_scores: Static base score overrides keyed by ``abs_id``.

    Returns:
        Integer priority score (higher wins collisions in ``_reverse_index``).
    """
    score = priority_scores.get(abs_id, 0) + (_TIER_SCORE_OFFSETS.get(tier, 0) if tier is not None else 0)
    raw_vars_val = details.get("variants")
    raw_vars = raw_vars_val if isinstance(raw_vars_val, dict) else {}
    return int(score + len(raw_vars))

  @classmethod
  def _register_api_candidates(
    cls,
    manager: "SemanticsManager",
    api_name: str,
    abstract_id: str,
    details: dict[str, JsonValue],
    score: int,
    alias_map: typing.Mapping[str, str],
    priority_scores: typing.Mapping[str, int],
  ) -> None:
    """Register ``api_name`` and its alias expansions into ``manager._reverse_index``.

    Args:
        manager: Target ``SemanticsManager`` instance.
        api_name: Concrete API path string from a variant entry.
        abstract_id: Abstract operation identifier.
        details: Operation definition dictionary.
        score: Precomputed priority score for ``abstract_id``.
        alias_map: Mapping from short import aliases to canonical module paths.
        priority_scores: Static base score overrides.
    """
    candidates = [api_name]
    parts = api_name.split(".")
    if parts[0] in alias_map:
      candidates.append(alias_map[parts[0]] + "." + ".".join(parts[1:]))
    for alias_name, module_path in alias_map.items():
      if api_name.startswith(module_path + "."):
        candidates.append(alias_name + api_name[len(module_path) :])

    try:
      op_model: OperationDef | None = manager.operations.get(abstract_id)
    except AttributeError:
      op_model = None

    for cand in candidates:
      if cand in manager._reverse_index:
        existing_id, existing_details = manager._reverse_index[cand]
        existing_tier = manager._key_origins.get(existing_id)
        existing_score = cls._get_priority(existing_id, existing_details, existing_tier, priority_scores)
        if score > existing_score:
          manager._reverse_index[cand] = (abstract_id, details)
          if op_model is not None:
            manager._typed_reverse_index[cand] = (abstract_id, op_model)
      else:
        manager._reverse_index[cand] = (abstract_id, details)
        if op_model is not None:
          manager._typed_reverse_index[cand] = (abstract_id, op_model)

  @classmethod
  def build_index(cls, manager: "SemanticsManager") -> None:
    """Construct the reverse index mapping from concrete API endpoints back to their abstract definitions.

    Args:
        manager: The SemanticsManager instance whose ``_reverse_index`` and ``_variant_cache`` will be populated.
    """
    cls._reset_caches(manager)
    alias_map = cls._load_alias_map(manager.framework_configs)
    priority_scores = cls._load_priority_scores()

    for abstract_id, details in manager.data.items():
      variants = details.get("variants", {})
      if not isinstance(variants, dict):
        continue
      tier = manager._key_origins.get(abstract_id)
      score = cls._get_priority(abstract_id, details, tier, priority_scores)

      for impl in variants.values():
        if not isinstance(impl, dict):
          continue
        api_name = impl.get("api")
        if isinstance(api_name, str) and api_name:
          cls._register_api_candidates(manager, api_name, abstract_id, details, score, alias_map, priority_scores)
