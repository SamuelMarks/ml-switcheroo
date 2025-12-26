"""
Tests for Verification Gated Rewriting.

Verifies that:
1. ASTEngine loads validation reports from config.
2. SemanticsManager stores and queries validation status.
3. PivotRewriter skips mappings that are marked as 'False' in the report.
4. Failed verifications are wrapped in EscapeHatches with specific messages.
"""

import json
import pytest
from typing import Set, Dict, Tuple, Optional, Any

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """
  Mock Manager with pre-loaded definitions for testing gating.
  """

  def __init__(self):
    self.data = {}
    self._validation_status = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._known_rng_methods = set()

    # New attributes
    self._providers = {}
    self._source_registry = {}

    # 1. 'good_op': Valid mapping, will pass verification
    self._inject("good_op", "torch.good", "jax.good")

    # 2. 'bad_op': Valid mapping structurally, but functionally FAIL
    self._inject("bad_op", "torch.bad", "jax.bad")

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def _inject(self, name, s_api, t_api):
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    return {}

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    return {}


@pytest.fixture
def mock_report(tmp_path):
  """Creates a validation report JSON file."""
  report = {"good_op": True, "bad_op": False}
  path = tmp_path / "verification.json"
  path.write_text(json.dumps(report))
  return path


def test_validation_gating_logic(mock_report):
  """
  Scenario:
      - `good_op` is True in report.
      - `bad_op` is False in report.
  Expectation:
      - `good_op` is rewritten to `jax.good`.
      - `bad_op` is skipped via EscapeHatch because of verification failure.
  """

  # 1. Setup Config with Report
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)

  # 2. Initialize Engine (which loads semantics and report)
  semantics = MockSemantics()
  engine = ASTEngine(semantics=semantics, config=config)

  # 3. Verify Report Loading
  assert semantics.is_verified("good_op") is True
  assert semantics.is_verified("bad_op") is False

  # 4. Run Transformation
  code = """
y1 = torch.good(x)
y2 = torch.bad(x)
"""
  result = engine.run(code)

  # Check Good Op (Rewritten)
  assert "jax.good(x)" in result.code

  # Check Bad Op (Preserved due to validation failure)
  assert "torch.bad(x)" in result.code

  # Check Failure Markers
  assert EscapeHatch.START_MARKER in result.code
  assert "Skipped 'torch.bad': Marked unsafe by verification report" in result.code


def test_missing_report_logic():
  """
  Scenario: No validation report provided.
  Expectation: Optimistic behavior (everything is allowed).
  """
  semantics = MockSemantics()
  # No report in config
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  engine = ASTEngine(semantics=semantics, config=config)

  code = "res = torch.bad(x)"
  result = engine.run(code)

  # Should rewrite despite being 'bad' in our hypothetical scenarios above,
  # because without a report, semantics says it's valid.
  assert "jax.bad(x)" in result.code
  assert EscapeHatch.START_MARKER not in result.code


def test_untracked_op_defaults_true(mock_report):
  """
  Scenario: Op matches semantics but is NOT in the report (untested).
  Expectation: Treated as Valid/Verified (Fail Open).
  """
  semantics = MockSemantics()
  semantics._inject("new_op", "torch.new", "jax.new")

  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=mock_report)

  engine = ASTEngine(semantics=semantics, config=config)

  code = "res = torch.new(x)"
  result = engine.run(code)

  assert "jax.new(x)" in result.code
