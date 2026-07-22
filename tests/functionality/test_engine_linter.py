"""Test suite for the Engine Linter module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from unittest.mock import MagicMock, patch
import libcst as cst


class MockUsageScanner(cst.CSTVisitor):
  """Mock Usage Scanner class for testing purposes."""

  def __init__(self, *args, **kwargs):
    """Initializes the MockUsageScanner instance."""
    pass

  def get_result(self):
    """Mock implementation of get result."""
    return True

  def on_visit(self, node):
    """Mock implementation of on visit."""
    return False

  def on_leave(self, node):
    """Mock implementation of on leave."""
    pass


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
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

  def get_import_map(self, target_fw):
    """Mock implementation of get import map."""
    return {}

  def get_framework_aliases(self):
    """Mock implementation of get framework aliases."""
    return {}

  def get_all_rng_methods(self):
    """Mock implementation of get all rng methods."""
    return set()

  def get_framework_config(self, fw):
    """Mock implementation of get framework configuration."""
    return {}


@pytest.fixture
def engine():
  """Provides a mock engine for testing."""
  mgr = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  mock_torch = MagicMock()
  mock_torch.configure_mock(import_alias=("torch", "torch"), inherits_from=None)
  del mock_torch.create_emitter
  del mock_torch.create_parser
  mock_jax = MagicMock()
  mock_jax.configure_mock(import_alias=("jax.numpy", "jnp"), inherits_from=None)
  del mock_jax.create_emitter
  del mock_jax.create_parser

  def get_adapter_side_effect(name):
    """Gets adapter side effect."""
    if name == "torch":
      return mock_torch
    if name == "jax":
      return mock_jax
    return None

  with patch("ml_switcheroo.frameworks.get_adapter", side_effect=get_adapter_side_effect):
    yield ASTEngine(semantics=mgr, config=config)


def test_engine_catches_leaked_import(engine):
  """Verifies the behavior of engine catches leaked import."""
  code = "\nimport torch\nx = 1\n"
  with patch("ml_switcheroo.core.engine.UsageScanner", side_effect=MockUsageScanner):
    result = engine.run(code)
  assert result.success is True
  assert len(result.errors) > 0
  assert any(("Forbidden Import: 'torch'" in e for e in result.errors))


def test_engine_catches_leaked_usage(engine):
  """Verifies the behavior of engine catches leaked usage."""
  code = "\nimport torch\ny = torch.abs(x)\n"
  result = engine.run(code)
  assert "torch.abs(x)" in result.code
  assert result.has_errors
  errors_str = str(result.errors)
  assert "Forbidden" in errors_str


def test_linter_trace_event(engine):
  """Verifies the behavior of linter trace event."""
  code = "import torch"
  with patch("ml_switcheroo.core.engine.UsageScanner", side_effect=MockUsageScanner):
    result = engine.run(code)
  phase_descriptions = [e["description"] for e in result.trace_events if e["type"] == "phase_start"]
  assert "Structural Linter" in phase_descriptions
