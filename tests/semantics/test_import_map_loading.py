"""Test suite for the Import Map Loading module."""

from typing import Dict, Any, Tuple, Optional
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self._providers: Dict[str, Any] = {}
    self._source_registry: Dict[str, Any] = {}
    self.data: Dict[str, Any] = {}
    self.import_data: Dict[str, Any] = {}
    self.framework_configs: Dict[str, Any] = {}
    self._reverse_index: Dict[str, Any] = {}


def test_no_hardcoded_defaults() -> None:
  """Verifies the behavior of no hardcoded defaults."""
  mgr: MockSemantics = MockSemantics()
  assert "torch.nn" not in mgr._source_registry


def test_merged_json_data() -> None:
  """Verifies the behavior of merged JSON data."""
  mgr: SemanticsManager = SemanticsManager()
  mgr._source_registry["torch.custom_sub"] = ("torch", SemanticTier.EXTRAS)
  mgr._providers["jax"] = {SemanticTier.EXTRAS: {"root": "my_lib", "sub": "mod", "alias": "cust"}}
  mapping: Dict[str, Tuple[str, Optional[str], Optional[str]]] = mgr.get_import_map(target_fw="jax")
  assert "torch.custom_sub" in mapping
  (root, sub, alias) = mapping["torch.custom_sub"]
  assert root == "my_lib"
  assert alias == "cust"


def test_get_import_map_structure() -> None:
  """Gets import map structure."""
  mgr: MockSemantics = MockSemantics()
  mgr._providers["jax"] = {
    SemanticTier.NEURAL: {"root": "flax", "sub": "linen", "alias": "nn"},
    SemanticTier.EXTRAS: {"root": "optax", "sub": None, "alias": None},
  }
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
  mgr._source_registry["torch.optim"] = ("torch", SemanticTier.EXTRAS)
  mapping: Dict[str, Tuple[str, Optional[str], Optional[str]]] = mgr.get_import_map(target_fw="jax")
  assert "torch.nn" in mapping
  val: Tuple[str, Optional[str], Optional[str]] = mapping["torch.nn"]
  assert isinstance(val, tuple)
  assert len(val) == 3
  assert val == ("flax", "linen", "nn")
  assert "torch.optim" in mapping
  val_optim: Tuple[str, Optional[str], Optional[str]] = mapping["torch.optim"]
  assert val_optim == ("optax", None, None)


def test_get_import_map_ignoring_irrelevant_targets() -> None:
  """Gets import map ignoring irrelevant targets."""
  mgr: MockSemantics = MockSemantics()
  mgr._source_registry["torch.stuff"] = ("torch", SemanticTier.EXTRAS)
  mgr._providers["tensorflow"] = {SemanticTier.EXTRAS: {"root": "tf", "sub": "stuff", "alias": None}}
  mapping: Dict[str, Tuple[str, Optional[str], Optional[str]]] = mgr.get_import_map(target_fw="jax")
  assert "torch.stuff" not in mapping
