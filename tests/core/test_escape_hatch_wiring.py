"""Test suite for the Escape Hatch Wiring module."""

import pytest
from pydantic import JsonValue

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` used in escape hatch wiring tests."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {
      "abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "DataLoader": {"std_args": ["dataset"], "variants": {"torch": {"api": "torch.utils.data.DataLoader"}, "jax": None}},
    }
    self.framework_configs = {}
    self._providers = {}
    self._source_registry = {}
    self._known_rng_methods: set[str] = {"seed", "manual_seed"}
    self._reverse_index = {
      "torch.abs": ("abs", self.data["abs"]),
      "torch.utils.data.DataLoader": ("DataLoader", self.data["DataLoader"]),
    }

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return self._known_rng_methods

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, target_framework: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        target_framework: Input `target_framework` parameter value (`str`).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    defn = self.data.get(abstract_id)
    if not defn:
      return None
    variants = defn.get("variants")
    if not isinstance(variants, dict):
      return None
    target = variants.get(target_framework)
    return target if isinstance(target, dict) else None

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Input `_id` parameter value (`str`).

    Returns:
        Resulting `bool` from is verified.
    """
    return True

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Input `target_framework` parameter value (`str`).

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    return {}


@pytest.fixture
def semantics_mgr() -> MockSemantics:
  """Create a configured `MockSemantics` fixture for escape hatch wiring tests.

  Returns:
      Configured `semantics_mgr` fixture value (MockSemantics).
  """
  return MockSemantics()


def test_escape_hatch_tier_c_gap(semantics_mgr: MockSemantics) -> None:
  """Verify `ASTEngine.from_options` emits `loader =` for escape hatch tier c gap.

  Args:
      semantics_mgr: Input `semantics_mgr` parameter value (`MockSemantics`).
  """
  engine = ASTEngine.from_options(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code: str = "loader = torch.utils.data.DataLoader(ds)"
  result: ConversionResult = engine.run(code)
  assert "torch.utils.data.DataLoader(ds)" in result.code
  assert "loader =" in result.code
  assert result.errors is not None
  assert len(result.errors) >= 1
  assert "Escape Hatches Detected" in result.errors[0]


def test_strict_mode_unknown_source_api(semantics_mgr: MockSemantics) -> None:
  """Verify strict mode unknown source API works as expected.

  Args:
      semantics_mgr: Input `semantics_mgr` parameter value (`MockSemantics`).
  """
  engine = ASTEngine.from_options(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code: str = "y = torch.weird_custom_func(x)"
  result: ConversionResult = engine.run(code)
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is True


def test_strict_mode_ignores_standard_python(semantics_mgr: MockSemantics) -> None:
  """Verify `ASTEngine.from_options` handles strict mode ignores standard python as expected.

  Args:
      semantics_mgr: Input `semantics_mgr` parameter value (`MockSemantics`).
  """
  engine = ASTEngine.from_options(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code: str = "z = len(x)"
  result: ConversionResult = engine.run(code)
  assert EscapeHatch.START_MARKER not in result.code
  assert "z = len(x)" in result.code
  assert result.has_errors is False


def test_default_mode_passthrough(semantics_mgr: MockSemantics) -> None:
  """Verify `ASTEngine.from_options` emits `torch.weird_custom_func(x)` for default mode passthrough.

  Args:
      semantics_mgr: Input `semantics_mgr` parameter value (`MockSemantics`).
  """
  engine = ASTEngine.from_options(semantics=semantics_mgr, source="torch", target="jax", strict_mode=False)
  code: str = "y = torch.weird_custom_func(x)"
  result: ConversionResult = engine.run(code)
  assert EscapeHatch.START_MARKER not in result.code
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is False
