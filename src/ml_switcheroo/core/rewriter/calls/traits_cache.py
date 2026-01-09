"""
Traits Caching Logic.

Provides lazy-loading accessor methods for framework configuration via the Context.
"""

from typing import Set, Tuple, Optional, TYPE_CHECKING

from ml_switcheroo.semantics.schema import StructuralTraits

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.calls.mixer import ApiStage


class TraitsCachingMixin:
  """
  Mixin for lazily loading and caching framework traits.

  Attributes:
      _cached_source_traits: Cached traits for source.
      _cached_target_traits: Cached traits for target.
  """

  _cached_source_traits: Optional[StructuralTraits] = None
  _cached_target_traits: Optional[StructuralTraits] = None

  def _get_source_traits(self: "ApiStage") -> StructuralTraits:
    """
    Lazily loads and caches the StructuralTraits of the SOURCE framework.
    """
    if self._cached_source_traits:
      return self._cached_source_traits

    config_dict = self.context.semantics.get_framework_config(self.context.source_fw)
    if config_dict and "traits" in config_dict:
      self._cached_source_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_source_traits = StructuralTraits()
    return self._cached_source_traits

  def _get_source_lifecycle_lists(self: "ApiStage") -> Tuple[Set[str], Set[str]]:
    """
    Lazily loads the lifecycle strip/warn lists from the SOURCE framework config.
    """
    traits = self._get_source_traits()
    return (
      set(traits.lifecycle_strip_methods),
      set(traits.lifecycle_warn_methods),
    )

  def _get_target_traits(self: "ApiStage") -> StructuralTraits:
    """
    Lazily loads properties of the TARGET framework.
    """
    if self._cached_target_traits:
      return self._cached_target_traits

    config_dict = self.context.semantics.get_framework_config(self.context.target_fw)
    if config_dict and "traits" in config_dict:
      self._cached_target_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_target_traits = StructuralTraits()

    return self._cached_target_traits
