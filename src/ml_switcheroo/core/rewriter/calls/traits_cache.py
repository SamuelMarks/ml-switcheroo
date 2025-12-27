"""
Traits Caching Logic.

This module provides the ``TraitsCachingMixin``, which implements a caching layer
for retrieving framework configuration objects. This avoids repeated expensive lookups
in the SemanticsManager during AST traversal.
"""

from typing import Set, Tuple

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.semantics.schema import StructuralTraits


class TraitsCachingMixin(BaseRewriter):
  """
  Mixin for lazily loading and caching framework traits.

  Attributes:
      _cached_source_traits: Cached `StructuralTraits` object for the source framework.
      _cached_target_traits: Cached `StructuralTraits` object for the target framework.
  """

  # Internal cache for traits to avoid lookup overhead per call
  _cached_source_traits: StructuralTraits = None
  _cached_target_traits: StructuralTraits = None

  def _get_source_traits(self) -> StructuralTraits:
    """
    Lazily loads and caches the StructuralTraits of the SOURCE framework.

    Returns:
        StructuralTraits: Configuration object for the source framework.
    """
    if self._cached_source_traits:
      return self._cached_source_traits

    config_dict = self.semantics.get_framework_config(self.source_fw)
    if config_dict and "traits" in config_dict:
      self._cached_source_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_source_traits = StructuralTraits()
    return self._cached_source_traits

  def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
    """
    Lazily loads the lifecycle strip/warn lists from the SOURCE framework config.

    Returns:
        Tuple[Set[str], Set[str]]: A tuple containing (strip_methods, warn_methods).
    """
    traits = self._get_source_traits()
    return (
      set(traits.lifecycle_strip_methods),
      set(traits.lifecycle_warn_methods),
    )

  def _get_target_traits(self) -> StructuralTraits:
    """
    Lazily loads properties of the TARGET framework.

    Returns:
        StructuralTraits: Configuration object for the target framework.
    """
    if self._cached_target_traits:
      return self._cached_target_traits

    config_dict = self.semantics.get_framework_config(self.target_fw)
    if config_dict and "traits" in config_dict:
      self._cached_target_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_target_traits = StructuralTraits()

    return self._cached_target_traits
