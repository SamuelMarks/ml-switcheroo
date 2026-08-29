"""Test suite for the Verification Gating module."""

import json
import typing
from pathlib import Path

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
    self._key_origins: dict[str, str] = {}
    self._inject("good_op", "torch.good", "jax.good")
    self._inject("bad_op", "torch.bad", "jax.bad")

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def _inject(self, name: str, s_api: str, t_api: str) -> None:
    """Mock implementation of  inject."""
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map."""
    return {}

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return {}

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get_definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve_variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified using loaded validation status."""
    return self._validation_status.get(_id, True)


@pytest.fixture
def mock_report(tmp_path: Path) -> str:
  """Docstring."""
  report: dict[str, bool] = {"good_op": True, "bad_op": False}
  path: Path = tmp_path / "verification.json"
  path.write_text(json.dumps(report))
  return str(path)


def test_validation_gating_logic(mock_report: str) -> None:
  """Verifies the behavior of validation gating logic."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)
  semantics = MockSemantics()
  semantics.load_validation_report(Path(mock_report))
  engine = ASTEngine(semantics=semantics, config=config)
  assert semantics.is_verified("good_op") is True
  assert semantics.is_verified("bad_op") is False
  code: str = "\ny1 = torch.good(x)\ny2 = torch.bad(x)\n"
  result: ConversionResult = engine.run(code)
  assert "jax.good(x)" in result.code
  assert "torch.bad(x)" in result.code
  assert EscapeHatch.START_MARKER in result.code
  assert "Skipped 'torch.bad': Marked unsafe by verification report" in result.code


def test_missing_report_logic() -> None:
  """Verifies the behavior of missing report logic."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code: str = "res = torch.bad(x)"
  result: ConversionResult = engine.run(code)
  assert "jax.bad(x)" in result.code
  assert EscapeHatch.START_MARKER not in result.code


def test_untracked_op_defaults_true(mock_report: str) -> None:
  """Verifies the behavior of untracked op defaults true."""
  semantics = MockSemantics()
  semantics._inject("new_op", "torch.new", "jax.new")
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)
  semantics.load_validation_report(Path(mock_report))
  engine = ASTEngine(semantics=semantics, config=config)
  code: str = "res = torch.new(x)"
  result: ConversionResult = engine.run(code)
  assert "jax.new(x)" in result.code
