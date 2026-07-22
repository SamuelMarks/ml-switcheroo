"""Test suite for the Verification Gating module."""

import json
import pytest
from typing import Set, Dict, Tuple, Optional, Any
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._validation_status = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._known_rng_methods = set()
    self._providers = {}
    self._source_registry = {}
    self._inject("good_op", "torch.good", "jax.good")
    self._inject("bad_op", "torch.bad", "jax.bad")

  def get_all_rng_methods(self) -> Set[str]:
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def _inject(self, name, s_api, t_api):
    """Mock implementation of  inject."""
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Mock implementation of get import map."""
    return {}

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Mock implementation of get framework configuration."""
    return {}


@pytest.fixture
def mock_report(tmp_path):
  """Provides a mock report for testing."""
  report = {"good_op": True, "bad_op": False}
  path = tmp_path / "verification.json"
  path.write_text(json.dumps(report))
  return path


def test_validation_gating_logic(mock_report):
  """Verifies the behavior of validation gating logic."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)
  semantics = MockSemantics()
  engine = ASTEngine(semantics=semantics, config=config)
  assert semantics.is_verified("good_op") is True
  assert semantics.is_verified("bad_op") is False
  code = "\ny1 = torch.good(x)\ny2 = torch.bad(x)\n"
  result = engine.run(code)
  assert "jax.good(x)" in result.code
  assert "torch.bad(x)" in result.code
  assert EscapeHatch.START_MARKER in result.code
  assert "Skipped 'torch.bad': Marked unsafe by verification report" in result.code


def test_missing_report_logic():
  """Verifies the behavior of missing report logic."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "res = torch.bad(x)"
  result = engine.run(code)
  assert "jax.bad(x)" in result.code
  assert EscapeHatch.START_MARKER not in result.code


def test_untracked_op_defaults_true(mock_report):
  """Verifies the behavior of untracked op defaults true."""
  semantics = MockSemantics()
  semantics._inject("new_op", "torch.new", "jax.new")
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)
  engine = ASTEngine(semantics=semantics, config=config)
  code = "res = torch.new(x)"
  result = engine.run(code)
  assert "jax.new(x)" in result.code
