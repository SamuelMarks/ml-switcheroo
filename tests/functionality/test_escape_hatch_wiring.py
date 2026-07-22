"""Test suite for the Escape Hatch Wiring module."""

import pytest
from typing import Set, Dict, Tuple, Optional
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {
      "abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "DataLoader": {"std_args": ["dataset"], "variants": {"torch": {"api": "torch.utils.data.DataLoader"}, "jax": None}},
    }
    self.framework_configs = {}
    self._providers = {}
    self._source_registry = {}
    self._known_rng_methods = {"seed", "manual_seed"}
    self._reverse_index = {
      "torch.abs": ("abs", self.data["abs"]),
      "torch.utils.data.DataLoader": ("DataLoader", self.data["DataLoader"]),
    }

  def get_all_rng_methods(self) -> Set[str]:
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def get_definition(self, name):
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id, target_fw):
    """Mock implementation of resolve variant."""
    defn = self.data.get(abstract_id)
    if not defn:
      return None
    return defn["variants"].get(target_fw)

  def is_verified(self, _id):
    """Mock implementation of is verified."""
    return True

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Mock implementation of get import map."""
    return {}


@pytest.fixture
def semantics_mgr():
  """Provides a mock semantics mgr for testing."""
  return MockSemantics()


def test_escape_hatch_tier_c_gap(semantics_mgr):
  """Verifies the behavior of escape hatch tier c gap."""
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code = "loader = torch.utils.data.DataLoader(ds)"
  result = engine.run(code)
  assert "torch.utils.data.DataLoader(ds)" in result.code
  assert "loader =" in result.code
  assert len(result.errors) >= 1
  assert "Escape Hatches Detected" in result.errors[0]


def test_strict_mode_unknown_source_api(semantics_mgr):
  """Verifies the behavior of strict mode unknown source API."""
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code = "y = torch.weird_custom_func(x)"
  result = engine.run(code)
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is True


def test_strict_mode_ignores_standard_python(semantics_mgr):
  """Verifies the behavior of strict mode ignores standard python."""
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code = "z = len(x)"
  result = engine.run(code)
  assert EscapeHatch.START_MARKER not in result.code
  assert "z = len(x)" in result.code
  assert result.has_errors is False


def test_default_mode_passthrough(semantics_mgr):
  """Verifies the behavior of default mode passthrough."""
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=False)
  code = "y = torch.weird_custom_func(x)"
  result = engine.run(code)
  assert EscapeHatch.START_MARKER not in result.code
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is False
