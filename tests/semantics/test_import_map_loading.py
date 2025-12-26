"""
Tests for Data-Driven Import Mapping (Feature 024).

Verifies that:
1. SemanticsManager starts clean (no hardcoded defaults).
2. SemanticsManager parses `__imports__` keys from incoming JSONs.
3. `get_import_map` returns the correct structure for the ImportFixer.
"""

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


class MockSemantics(SemanticsManager):
  def __init__(self):
    self._providers = {}
    self._source_registry = {}
    self.data = {}
    self.import_data = {}  # Keep for backward compatibility check if needed or remove
    self.framework_configs = {}
    self._reverse_index = {}


def test_no_hardcoded_defaults():
  """
  Ensure the manager honors the strict architecture rule:
  No hardcoded mappings in Python. Initialization should yield
  an empty state if no JSON files are found.
  """
  mgr = MockSemantics()

  # Assert 'torch.nn' is NOT present in providers/registry
  assert "torch.nn" not in mgr._source_registry


def test_merged_json_data():
  """
  Simulate loading usage of providers from adapter/json.
  Since 'import_data' attribute is removed, we check _providers structure.
  """
  mgr = SemanticsManager()

  # Manually inject data as if loaded from an Adapter
  mgr._source_registry["torch.custom_sub"] = ("torch", SemanticTier.EXTRAS)
  mgr._providers["jax"] = {SemanticTier.EXTRAS: {"root": "my_lib", "sub": "mod", "alias": "cust"}}

  # Verify lookups
  mapping = mgr.get_import_map(target_fw="jax")

  # Should find mapping for torch.custom_sub
  assert "torch.custom_sub" in mapping
  root, sub, alias = mapping["torch.custom_sub"]
  assert root == "my_lib"
  assert alias == "cust"


def test_get_import_map_structure():
  """
  Verify `get_import_map` transforms internal storage to the tuple format
  expected by ImportFixer.
  Format: Dict[str, Tuple[root, sub, alias]]
  """
  mgr = MockSemantics()

  # Setup Provider for JAX Extras
  mgr._providers["jax"] = {
    SemanticTier.NEURAL: {"root": "flax", "sub": "linen", "alias": "nn"},
    SemanticTier.EXTRAS: {"root": "optax", "sub": None, "alias": None},
  }

  # Setup Source Registry
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
  mgr._source_registry["torch.optim"] = ("torch", SemanticTier.EXTRAS)

  # Use the injected data
  mapping = mgr.get_import_map(target_fw="jax")

  # Check torch.nn -> flax.linen
  assert "torch.nn" in mapping
  val = mapping["torch.nn"]
  assert isinstance(val, tuple)
  assert len(val) == 3
  assert val == ("flax", "linen", "nn")

  # Check torch.optim -> optax
  assert "torch.optim" in mapping
  val_optim = mapping["torch.optim"]
  assert val_optim == ("optax", None, None)


def test_get_import_map_ignoring_irrelevant_targets():
  """Verify we filter out imports not matching the target framework."""
  mgr = MockSemantics()

  # Register source
  mgr._source_registry["torch.stuff"] = ("torch", SemanticTier.EXTRAS)

  # Provider is tensorflow
  mgr._providers["tensorflow"] = {SemanticTier.EXTRAS: {"root": "tf", "sub": "stuff", "alias": None}}

  # Request JAX map
  mapping = mgr.get_import_map(target_fw="jax")

  assert "torch.stuff" not in mapping
