"""Test suite for the Verification Gating module."""

import json
from pathlib import Path

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, dict[str, JsonValue]] = {}
    self._validation_status: dict[str, bool] = {}
    self.framework_configs: dict[str, dict[str, JsonValue]] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, dict[SemanticTier | str, dict[str, str | None]]] = {}
    self._source_registry: dict[str, tuple[str, SemanticTier | str]] = {}
    self._key_origins: dict[str, str] = {}
    self._inject("good_op", "torch.good", "jax.good")
    self._inject("bad_op", "torch.bad", "jax.bad")

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return self._known_rng_methods

  def _inject(self, name: str, s_api: str, t_api: str) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
    """
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get_definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve_variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    variants = self.data.get(abstract_id, {}).get("variants")
    if isinstance(variants, dict):
      res = variants.get(fw)
      if isinstance(res, dict):
        return res
    return None

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified using loaded validation status.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._validation_status.get(_id, True)


@pytest.fixture
def mock_report(tmp_path: Path) -> str:
  """Execute mock report helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  report: dict[str, bool] = {"good_op": True, "bad_op": False}
  path = tmp_path / "verification.json"
  path.write_text(json.dumps(report))
  return str(path)


def test_validation_gating_logic(mock_report: str) -> None:
  """Verifies the behavior of validation gating logic.

  Args:
      mock_report: Test argument or fixture.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=Path(mock_report))
  semantics = MockSemantics()
  semantics.load_validation_report(Path(mock_report))
  engine = ASTEngine(semantics=semantics, config=config)
  assert semantics.is_verified("good_op") is True
  assert semantics.is_verified("bad_op") is False
  code = """
y1 = torch.good(x)
y2 = torch.bad(x)
"""
  result = engine.run(code)
  assert "jax.good(x)" in result.code
  assert "torch.bad(x)" in result.code
  assert EscapeHatch.START_MARKER in result.code
  assert "Skipped 'torch.bad': Marked unsafe by verification report" in result.code


def test_missing_report_logic() -> None:
  """Verifies the behavior of missing report logic."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "res = torch.bad(x)"
  result = engine.run(code)
  assert "jax.bad(x)" in result.code
  assert EscapeHatch.START_MARKER not in result.code


def test_untracked_op_defaults_true(mock_report: str) -> None:
  """Verifies the behavior of untracked op defaults true.

  Args:
      mock_report: Test argument or fixture.
  """
  semantics = MockSemantics()
  semantics._inject("new_op", "torch.new", "jax.new")
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=Path(mock_report))
  semantics.load_validation_report(Path(mock_report))
  engine = ASTEngine(semantics=semantics, config=config)
  code = "res = torch.new(x)"
  result = engine.run(code)
  assert "jax.new(x)" in result.code
