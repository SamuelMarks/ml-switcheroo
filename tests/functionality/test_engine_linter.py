"""
Tests for Engine <-> Linter Integration.

Verifies that:
1. ASTEngine automatically calls StructuralLinter at the end of run().
2. Leaked artifacts (imports, aliases) result in errors in ConversionResult.
3. Trace logs capture linter warnings.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.scanners import UsageScanner
from unittest.mock import MagicMock, patch
import libcst as cst


# Define a concrete mock class to satisfy LibCST's visitor requirements
# Must explicitly inherit CSTVisitor to pass isinstance check in LibCST runner
class MockUsageScanner(cst.CSTVisitor):
  def __init__(self, *args, **kwargs):
    pass

  def get_result(self):
    return True

  def on_visit(self, node):
    return False

  def on_leave(self, node):
    pass


class MockSemantics(SemanticsManager):
  """Minimal semantics to handle basic ops."""

  def __init__(self):
    # Use super's structure but empty data
    self.data = {}
    self.framework_configs = {}
    self.import_data = {}
    self.test_templates = {}
    self._known_rng_methods = set()
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._providers = {}
    self._source_registry = {}

  # Valid stub for engine usage
  def get_import_map(self, target_fw):
    return {}

  def get_framework_aliases(self):
    return {}

  def get_all_rng_methods(self):
    return set()

  def get_framework_config(self, fw):
    return {}


@pytest.fixture
def engine():
  mgr = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)

  # Setup robust mocks for adapters
  mock_torch = MagicMock()
  mock_torch.configure_mock(import_alias=("torch", "torch"), inherits_from=None)

  mock_jax = MagicMock()
  mock_jax.configure_mock(import_alias=("jax.numpy", "jnp"), inherits_from=None)

  def get_adapter_side_effect(name):
    if name == "torch":
      return mock_torch
    if name == "jax":
      return mock_jax
    return None

  # Patch the location where Engine imports it
  with patch("ml_switcheroo.frameworks.get_adapter", side_effect=get_adapter_side_effect):
    yield ASTEngine(semantics=mgr, config=config)


def test_engine_catches_leaked_import(engine):
  """
  Scenario: Input has `import torch`. Rewriter fails to remove it.
  Expectation: Engine returns result with Lint Violation errors.
  """
  code = """ 
import torch
x = 1
"""
  # Use MockUsageScanner class instead of instance patching to satisfy LibCST checks
  with patch("ml_switcheroo.core.engine.UsageScanner", side_effect=MockUsageScanner):
    result = engine.run(code)

  assert result.success is True
  assert len(result.errors) > 0
  assert any("Forbidden Import: 'torch'" in e for e in result.errors)


def test_engine_catches_leaked_usage(engine):
  """
  Scenario: Input has `torch.abs(x)`. No Semantics defined to map it.
  Rewriter passes it through (Lax Mode).
  Expectation: Linter catches 'torch.abs'.
  """
  code = """ 
import torch
y = torch.abs(x) 
"""
  # Here we don't force UsageScanner, we let it detect 'torch.abs' usage naturally.
  # The engine should preserve 'import torch' because 'torch' is used in 'torch.abs'.
  # The linter should then flag 'Forbidden Import' and 'Forbidden Usage'.

  result = engine.run(code)

  assert "torch.abs(x)" in result.code
  assert result.has_errors

  errors_str = str(result.errors)
  assert "Forbidden" in errors_str


def test_linter_trace_event(engine):
  """
  Verify linter execution is logged in trace.
  """
  code = "import torch"

  with patch("ml_switcheroo.core.engine.UsageScanner", side_effect=MockUsageScanner):
    result = engine.run(code)

  # Check for Structural Linter phase
  phases = [e["description"] for e in result.trace_events if e["type"] == "phase_start"]
  assert "Structural Linter" in phases
