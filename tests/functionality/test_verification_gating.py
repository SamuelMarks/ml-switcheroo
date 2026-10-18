"""Test suite for the Verification Gating module."""

import json
import typing
from pathlib import Path

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: typing.Any = {}
    self._validation_status: typing.Any = {}
    self.framework_configs: typing.Any = {}
    self.import_data: typing.Any = {}
    self._reverse_index: typing.Any = {}
    self._known_rng_methods: typing.Any = set()
    self._providers: typing.Any = {}
    self._source_registry: typing.Any = {}
    self._key_origins: typing.Any = {}
    self._inject("good_op", "torch.good", "jax.good")
    self._inject("bad_op", "torch.bad", "jax.bad")

  def get_all_rng_methods(self) -> typing.Any:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return self._known_rng_methods

  def _inject(self, name: typing.Any, s_api: typing.Any, t_api: typing.Any) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
    """
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_import_map(self, target_fw: typing.Any) -> typing.Any:
    """Mock implementation of get import map.

    Args:
        target_fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_framework_config(self, framework: typing.Any) -> typing.Any:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_definition(self, name: typing.Any) -> typing.Any:
    """Mock get_definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: typing.Any, fw: typing.Any) -> typing.Any:
    """Mock resolve_variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)

  def is_verified(self, _id: typing.Any) -> typing.Any:
    """Mock is_verified using loaded validation status.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._validation_status.get(_id, True)


@pytest.fixture
def mock_report(tmp_path: typing.Any) -> typing.Any:
  """Execute mock report helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  report: typing.Any = {"good_op": True, "bad_op": False}
  path: typing.Any = tmp_path / "verification.json"
  path.write_text(json.dumps(report))
  return str(path)


def test_validation_gating_logic(mock_report: typing.Any) -> None:
  """Verifies the behavior of validation gating logic.

  Args:
      mock_report: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)
  semantics: typing.Any = MockSemantics()
  semantics.load_validation_report(Path(mock_report))
  engine: typing.Any = ASTEngine(semantics=semantics, config=config)
  assert semantics.is_verified("good_op") is True
  assert semantics.is_verified("bad_op") is False
  code: typing.Any = "\ny1 = torch.good(x)\ny2 = torch.bad(x)\n"
  result: typing.Any = engine.run(code)
  assert "jax.good(x)" in result.code
  assert "torch.bad(x)" in result.code
  assert EscapeHatch.START_MARKER in result.code
  assert "Skipped 'torch.bad': Marked unsafe by verification report" in result.code


def test_missing_report_logic() -> None:
  """Verifies the behavior of missing report logic."""
  semantics: typing.Any = MockSemantics()
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine: typing.Any = ASTEngine(semantics=semantics, config=config)
  code: typing.Any = "res = torch.bad(x)"
  result: typing.Any = engine.run(code)
  assert "jax.bad(x)" in result.code
  assert EscapeHatch.START_MARKER not in result.code


def test_untracked_op_defaults_true(mock_report: typing.Any) -> None:
  """Verifies the behavior of untracked op defaults true.

  Args:
      mock_report: Test argument or fixture.
  """
  semantics: typing.Any = MockSemantics()
  semantics._inject("new_op", "torch.new", "jax.new")
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)
  semantics.load_validation_report(Path(mock_report))
  engine: typing.Any = ASTEngine(semantics=semantics, config=config)
  code: typing.Any = "res = torch.new(x)"
  result: typing.Any = engine.run(code)
  assert "jax.new(x)" in result.code
