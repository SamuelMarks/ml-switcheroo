"""Test suite for the Escape Hatch Wiring module."""

from pydantic import JsonValue

import pytest

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

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
        abstract_id: The abstract id (str) for resolve variant.
        target_framework: The target framework (str) for resolve variant.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    defn = self.data.get(abstract_id)
    if not defn:
      return None
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": defn}, "__getitem__")("v")["variants"]}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": target_framework}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: The  id (str) for is verified.

    Returns:
        Resulting `bool` from is verified.
    """
    return True

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: The target framework (str) for get import map.

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    return {}


@pytest.fixture
def semantics_mgr() -> MockSemantics:
  """Execute semantics mgr helper logic.

  Returns:
      Configured `semantics_mgr` fixture value (MockSemantics).
  """
  return MockSemantics()


def test_escape_hatch_tier_c_gap(semantics_mgr: MockSemantics) -> None:
  """Verifies the behavior of escape hatch tier c gap.

  Args:
      semantics_mgr: The semantics mgr (MockSemantics) for test escape hatch tier c gap.
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
      semantics_mgr: The semantics mgr (MockSemantics) for test strict mode unknown source api.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)
  code: str = "y = torch.weird_custom_func(x)"
  result: ConversionResult = engine.run(code)
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is True


def test_strict_mode_ignores_standard_python(semantics_mgr: MockSemantics) -> None:
  """Verifies the behavior of strict mode ignores standard python.

  Args:
      semantics_mgr: The semantics mgr (MockSemantics) for test strict mode ignores standard python.
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
      semantics_mgr: The semantics mgr (MockSemantics) for test default mode passthrough.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=False)
  code: str = "y = torch.weird_custom_func(x)"
  result: ConversionResult = engine.run(code)
  assert EscapeHatch.START_MARKER not in result.code
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is False
