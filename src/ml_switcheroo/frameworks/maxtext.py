"""MaxText Framework Adapter (Level 2).

This adapter specializes the core JAX stack for Google's MaxText framework.
It inherits Level 0 (Core JAX) and Level 1 (Optax/Orbax) capabilities from
``JAXStackMixin`` and provides structural traits for distributed sharding.
"""

from typing import Dict, List

from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardMap,
  ImportConfig,
)
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin
from ml_switcheroo_ir.schema.ghost import SemanticTier


@register_framework("maxtext")
class MaxTextAdapter(JAXStackMixin):
  """Adapter class for MaxText.

  MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax.
  """

  display_name: str = "MaxText"

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Return standard imports for MaxText.

    Returns:
        Dict mapping namespace paths to ImportConfig.
    """
    return {"maxtext": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="maxtext")}

  @property
  def structural_traits(self) -> StructuralTraits:
    """Structural rewriting traits for MaxText.

    Returns:
        StructuralTraits: Configuration object.
    """
    return StructuralTraits(
      module_base="maxtext.layers.Layer",
      forward_method="__call__",
      functional_execution_method="apply",
      jit_static_args=["axis", "axes", "dim", "dims", "keepdim", "keepdims", "dtype"],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Plugin capabilities indicating required behaviors in the target framework.

    Returns:
        PluginTraits: Configuration object.
    """
    return PluginTraits(
      supports_sharding=True,
      requires_rng_threading=True,
      requires_optimizer_unwrapping=True,
      supports_inplace_ops=False,
    )

  @property
  def harness_imports(self) -> List[str]:
    """Import for Harness generation."""
    return ["import maxtext"]

  def get_harness_init_code(self) -> str:
    """Logic to initialize maxtext for the harness."""
    return ""

  def get_tier_definitions(self, tier: SemanticTier) -> StandardMap:
    """Get mappings for the requested tier.

    Args:
        tier: The semantic tier to retrieve mappings for.

    Returns:
        A dictionary mapping abstract keys to framework-specific definitions.
    """
    return {}

  def discover_apis(self):
    """Dynamically discover MaxText APIs.

    Returns:
        A dictionary of dynamically discovered APIs.
    """
    return {}

  def verify_environment(self) -> bool:
    """Verify if maxtext is installed.

    Returns:
        True if the environment supports MaxText, False otherwise.
    """
    return True
