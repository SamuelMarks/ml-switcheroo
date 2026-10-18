"""Test suite for the Escape Hatch Wiring module."""

import typing

import pytest

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {
      "abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "DataLoader": {"std_args": ["dataset"], "variants": {"torch": {"api": "torch.utils.data.DataLoader"}, "jax": None}},
    }
    self.framework_configs: dict[str, typing.Any] = {}
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = {"seed", "manual_seed"}
    self._reverse_index: dict[str, typing.Any] = {
      "torch.abs": ("abs", self.data["abs"]),
      "torch.utils.data.DataLoader": ("DataLoader", self.data["DataLoader"]),
    }

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return self._known_rng_methods

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, target_fw: str) -> typing.Any:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        target_fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    defn: typing.Any = self.data.get(abstract_id)
    if not defn:
      return None
    return defn["variants"].get(target_fw)

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map.

    Args:
        target_fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


@pytest.fixture
def semantics_mgr() -> MockSemantics:
  """Execute semantics mgr helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return MockSemantics()


def test_escape_hatch_tier_c_gap(semantics_mgr: MockSemantics) -> None:
  """Verifies the behavior of escape hatch tier c gap.

  Args:
      semantics_mgr: Test argument or fixture.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code: str = "loader = torch.utils.data.DataLoader(ds)"
  result: ConversionResult = engine.run(code)
  assert "torch.utils.data.DataLoader(ds)" in result.code
  assert "loader =" in result.code
  assert result.errors is not None
  assert len(result.errors) >= 1
  assert "Escape Hatches Detected" in result.errors[0]


def test_strict_mode_unknown_source_api(semantics_mgr: MockSemantics) -> None:
  """Verifies the behavior of strict mode unknown source API.

  Args:
      semantics_mgr: Test argument or fixture.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code: str = "y = torch.weird_custom_func(x)"
  result: ConversionResult = engine.run(code)
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is True


def test_strict_mode_ignores_standard_python(semantics_mgr: MockSemantics) -> None:
  """Verifies the behavior of strict mode ignores standard python.

  Args:
      semantics_mgr: Test argument or fixture.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code: str = "z = len(x)"
  result: ConversionResult = engine.run(code)
  assert EscapeHatch.START_MARKER not in result.code
  assert "z = len(x)" in result.code
  assert result.has_errors is False


def test_default_mode_passthrough(semantics_mgr: MockSemantics) -> None:
  """Verifies the behavior of default mode passthrough.

  Args:
      semantics_mgr: Test argument or fixture.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=False)
  code: str = "y = torch.weird_custom_func(x)"
  result: ConversionResult = engine.run(code)
  assert EscapeHatch.START_MARKER not in result.code
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is False
