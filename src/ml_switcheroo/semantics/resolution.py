"""Multi-tier semantic operation resolution and variant dispatch logic."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.semantics.schema import OperationDataDict


@dataclass
class ResolutionContext:
  """Context object encapsulating source and target frameworks during semantic resolution."""

  source_framework: str
  target_framework: str
  current_scope: str = "global"


class SemanticResolver:
  """Lightweight resolver mapping framework API calls to abstract operations and target APIs."""

  def __init__(self, operation_maps: Dict[str, OperationDataDict]) -> None:
    """Initialize the resolver with an operation mapping dictionary.

    Args:
        operation_maps: Dictionary mapping abstract operation names to their variant definitions.
    """
    self.operation_maps = operation_maps

  def resolve(self, framework: str, api_call: str) -> Optional[str]:
    """Resolve a concrete framework API call string to its abstract operation name.

    Args:
        framework: Source framework key (e.g., 'torch').
        api_call: Concrete API string (e.g., 'torch.add').

    Returns:
        Matching abstract operation identifier if found, else None.
    """
    for abstract_op, mapping in self.operation_maps.items():
      variants = mapping.get("variants") or {}
      fw_variant = variants.get(framework) or {}
      if fw_variant.get("api") == api_call:
        return abstract_op
    return None

  def dispatch(self, abstract_op: str, target_framework: str) -> Optional[str]:
    """Dispatch an abstract operation name to the target framework's concrete API path.

    Args:
        abstract_op: Abstract operation identifier (e.g., 'Add').
        target_framework: Target framework key (e.g., 'jax').

    Returns:
        Target concrete API path if defined, else None.
    """
    mapping = self.operation_maps.get(abstract_op)
    if mapping is not None:
      variants = mapping.get("variants") or {}
      fw_variant = variants.get(target_framework) or {}
      api: Optional[str] = fw_variant.get("api")
      return api
    return None


class ResolutionEngine:
  """Modular resolution engine handling inheritance-aware variant lookup and reverse-index queries."""

  @staticmethod
  def resolve_variant(
    manager: SemanticsManager, abstract_id: str, target_fw: str
  ) -> typing.Optional[dict[str, typing.Any]]:
    """Resolve the implementation variant dictionary for ``abstract_id`` in ``target_fw``.

    Traverses framework inheritance chains (e.g., ``flax_nnx -> jax``) and case-insensitive
    operation aliases, caching results in ``manager._variant_cache``.

    Args:
        manager: SemanticsManager instance holding operation data and inheritance links.
        abstract_id: Abstract operation identifier to resolve.
        target_fw: Target framework key.

    Returns:
        Variant mapping dictionary if resolved, else None.
    """
    cache_key = (abstract_id, target_fw)
    if not hasattr(manager, "_variant_cache"):
      manager._variant_cache = {}
    if cache_key in manager._variant_cache:
      return manager._variant_cache[cache_key]

    defn = manager.data.get(abstract_id)
    if not defn and isinstance(abstract_id, str):
      defn = manager.data.get(abstract_id.lower()) or manager.data.get(abstract_id.capitalize())
    if not defn:
      manager._variant_cache[cache_key] = None
      return None
    variants = defn.get("variants", {})
    if target_fw in variants:
      result = variants[target_fw]
      manager._variant_cache[cache_key] = result
      v1: typing.Optional[dict[str, typing.Any]] = result
      return v1

    curr = target_fw
    limit = 5
    while limit > 0:
      parent = manager._resolve_inheritance(curr)
      if not parent:
        break
      if parent in variants:
        result = variants[parent]
        manager._variant_cache[cache_key] = result
        v2: typing.Optional[dict[str, typing.Any]] = result
        return v2
      curr = parent
      limit -= 1

    if isinstance(abstract_id, str):
      for candidate_id in [abstract_id.lower(), abstract_id.capitalize()]:
        if candidate_id != abstract_id:
          alt_defn = manager.data.get(candidate_id)
          if alt_defn:
            alt_vars = alt_defn.get("variants", {})
            if target_fw in alt_vars:
              result = alt_vars[target_fw]
              manager._variant_cache[cache_key] = result
              v3: typing.Optional[dict[str, typing.Any]] = result
              return v3

    manager._variant_cache[cache_key] = None
    return None

  @staticmethod
  def get_definition(
    manager: SemanticsManager, api_name: str
  ) -> typing.Optional[typing.Tuple[str, dict[str, typing.Any]]]:
    """Look up the `(abstract_id, details)` tuple for a concrete API name or abstract key.

    Args:
        manager: SemanticsManager instance holding `_reverse_index` and `data`.
        api_name: Concrete API path (e.g., 'torch.abs') or abstract operation ID.

    Returns:
        Tuple of `(abstract_id, operation_details_dict)` if found, else None.
    """
    if not hasattr(manager, "_reverse_index"):
      manager._reverse_index = {}
    res: typing.Optional[typing.Tuple[str, dict[str, typing.Any]]] = manager._reverse_index.get(api_name)
    if res:
      return res

    lower_name = api_name.lower()
    cap_name = api_name.capitalize()
    for candidate in [api_name, lower_name, cap_name]:
      if candidate in manager.data and manager.data[candidate].get("variants"):
        return (candidate, manager.data[candidate])

    for candidate in [api_name, lower_name, cap_name]:
      if candidate in manager.data:
        return (candidate, manager.data[candidate])

    return None
