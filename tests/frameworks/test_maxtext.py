"""Test suite for the MaxText Framework Adapter."""

import typing
from ml_switcheroo.frameworks.maxtext import MaxTextAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_maxtext_adapter() -> None:
  """Test basic properties of MaxTextAdapter."""
  adapter = MaxTextAdapter()

  assert adapter.display_name == "MaxText"
  assert adapter.verify_environment() is True

  imports: typing.Any = adapter.import_namespaces
  assert "maxtext" in imports
  assert imports["maxtext"].recommended_alias == "maxtext"

  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "maxtext.layers.Layer"
  assert traits.forward_method == "__call__"
  assert traits.functional_execution_method == "apply"

  ptraits: typing.Any = adapter.plugin_traits
  assert ptraits.supports_sharding is True
  assert ptraits.requires_rng_threading is True

  assert adapter.get_tier_definitions(SemanticTier.NEURAL) == {}
  assert adapter.discover_apis() == {}
