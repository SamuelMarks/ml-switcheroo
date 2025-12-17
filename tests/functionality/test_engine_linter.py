"""
Tests for Engine <-> Linter Integration.

Verifies that:
1. ASTEngine automatically calls StructuralLinter at the end of run().
2. Leaked artifacts (imports, aliases) result in errors in ConversionResult.
3. Trace logs capture linter warnings.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from unittest.mock import MagicMock, patch


class MockSemantics(SemanticsManager):
  """Minimal semantics to handle basic ops."""

  def __init__(self):
    super().__init__()
    self.data = {}
    self.framework_configs = {}
    # We don't load files, ensuring a "clean" state where most ops fail translation


@pytest.fixture
def engine():
  mgr = MockSemantics()
  # Configure so that 'torch' is Source and 'jax' is Target
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return ASTEngine(semantics=mgr, config=config)


def test_engine_catches_leaked_import(engine):
  """
  Scenario: Input has `import torch`. Rewriter fails to remove it (e.g. preservation logic error).
  Expectation: Engine returns result with Lint Violation errors.
  """
  code = """
import torch
x = 1
"""
  # Patched UsageScanner to force preservation of "torch" so ImportFixer keeps it
  with patch("ml_switcheroo.core.engine.UsageScanner.get_result", return_value=True):
    result = engine.run(code)

  assert result.success is True  # Engine ran successfully
  assert len(result.errors) > 0
  assert any("Forbidden Import: 'torch'" in e for e in result.errors)


def test_engine_catches_leaked_usage(engine):
  """
  Scenario: Input has `torch.abs(x)`. No Semantics defined.
  Rewriter passes it through (Lax Mode).
  Expectation: Linter catches 'torch.abs' as a forbidden attribute usage.
  """
  code = """
import torch
y = torch.abs(x)
"""
  # Lax mode allows pass-through. ImportFixer preserves imports if usage found.
  # Linter should scream about 'torch' being present in output targeting 'jax'.

  result = engine.run(code)

  assert "torch.abs(x)" in result.code
  assert result.has_errors

  # Check for usage error
  errors_str = str(result.errors)
  assert "Forbidden Usage" in errors_str or "Forbidden Import" in errors_str


def test_linter_trace_event(engine):
  """
  Verify linter execution is logged in trace.
  """
  code = "import torch"
  # Force preservation
  with patch("ml_switcheroo.core.engine.UsageScanner.get_result", return_value=True):
    result = engine.run(code)

  # Check for Structural Linter phase
  phases = [e["description"] for e in result.trace_events if e["type"] == "phase_start"]
  assert "Structural Linter" in phases
