"""Multi-tier semantic operation resolution and variant dispatch logic."""

from __future__ import annotations

import typing

import pydantic

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.semantics.schema import OperationDataDict


class ResolutionContext(pydantic.BaseModel):
  """Context object encapsulating source and target frameworks during semantic resolution."""

  model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

  source_framework: str
  target_framework: str
  current_scope: str = "global"

  def __init__(
    self,
    source_framework: str,
    target_framework: str,
    current_scope: str = "global",
  ) -> None:
    """Initialize ResolutionContext.

    Args:
        source_framework: Source framework key.
        target_framework: Target framework key.
        current_scope: Current semantic resolution scope.
    """
    super().__init__(
      source_framework=source_framework,
      target_framework=target_framework,
      current_scope=current_scope,
    )


class SemanticResolver:
  """Lightweight resolver mapping framework API calls to abstract operations and target APIs."""

  def __init__(
    self, operation_maps: typing.Dict[str, typing.Union[OperationDataDict, dict[str, pydantic.JsonValue]]]
  ) -> None:
    """Initialize the resolver with an operation mapping dictionary.

    Args:
        operation_maps: Dictionary mapping abstract operation names to their variant definitions.
    """
    self.operation_maps = operation_maps

  def resolve(self, framework: str, api_call: str) -> typing.Optional[str]:
    """Resolve a concrete framework API call string to its abstract operation name.

    Args:
        framework: Source framework key (e.g., 'torch').
        api_call: Concrete API string (e.g., 'torch.add').

    Returns:
        Matching abstract operation identifier if found, else None.
    """
    for abstract_op, mapping in self.operation_maps.items():
      raw_map = mapping.model_dump() if isinstance(mapping, pydantic.BaseModel) else mapping
      variants = raw_map.get("variants") or {}
      fw_variant = getattr(variants, "get")(framework)
      if isinstance(fw_variant, dict) and fw_variant.get("api") == api_call:
        return abstract_op
    return None

  def dispatch(self, abstract_op: str, target_framework: str) -> typing.Optional[str]:
    """Dispatch an abstract operation name to the target framework's concrete API path.

    Args:
        abstract_op: Abstract operation identifier (e.g., 'Add').
        target_framework: Target framework key (e.g., 'jax').

    Returns:
        Target concrete API path if defined, else None.
    """
    mapping = self.operation_maps.get(abstract_op)
    if mapping is not None:
      raw_map = mapping.model_dump() if isinstance(mapping, pydantic.BaseModel) else mapping
      variants = raw_map.get("variants") or {}
      fw_variant = getattr(variants, "get")(target_framework)
      if isinstance(fw_variant, dict):
        api = fw_variant.get("api")
        return str(api) if api is not None else None
    return None


class ResolutionEngine:
  """Modular resolution engine handling inheritance-aware variant lookup and reverse-index queries."""

  @staticmethod
  def resolve_variant(
    manager: SemanticsManager, abstract_id: str, target_fw: str
  ) -> typing.Optional[dict[str, pydantic.JsonValue]]:
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
    raw_variants = defn.get("variants")
    variants: dict[str, typing.Optional[dict[str, pydantic.JsonValue]]] = (
      raw_variants if isinstance(raw_variants, dict) else {}  # type: ignore[assignment]
    )
    if target_fw in variants:
      result = variants[target_fw]
      manager._variant_cache[cache_key] = result
      return result

    curr = target_fw
    limit = 5
    while limit > 0:
      parent = manager._resolve_inheritance(curr)
      if not parent:
        break
      if parent in variants:
        result = variants[parent]
        manager._variant_cache[cache_key] = result
        return result
      curr = parent
      limit -= 1

    if isinstance(abstract_id, str):
      for candidate_id in [abstract_id.lower(), abstract_id.capitalize()]:
        if candidate_id != abstract_id:
          alt_defn = manager.data.get(candidate_id)
          if alt_defn:
            alt_vars = alt_defn.get("variants")
            if isinstance(alt_vars, dict) and target_fw in alt_vars:
              alt_res = alt_vars[target_fw]
              res_dict: typing.Optional[dict[str, pydantic.JsonValue]] = alt_res if isinstance(alt_res, dict) else None
              manager._variant_cache[cache_key] = res_dict
              return res_dict

    manager._variant_cache[cache_key] = None
    return None

  @staticmethod
  def get_definition(
    manager: SemanticsManager, api_name: str
  ) -> typing.Optional[typing.Tuple[str, dict[str, pydantic.JsonValue]]]:
    """Look up the `(abstract_id, details)` tuple for a concrete API name or abstract key.

    Args:
        manager: SemanticsManager instance holding `_reverse_index` and `data`.
        api_name: Concrete API path (e.g., 'torch.abs') or abstract operation ID.

    Returns:
        Tuple of `(abstract_id, operation_details_dict)` if found, else None.
    """
    if not hasattr(manager, "_reverse_index"):
      manager._reverse_index = {}
    res: typing.Optional[typing.Tuple[str, dict[str, pydantic.JsonValue]]] = manager._reverse_index.get(api_name)
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
