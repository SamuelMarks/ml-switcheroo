"""
Tests for RewriterContext Mechanics.

Verifies:
1. Context instantiation and state initialization.
2. Property accessors for configuration.
3. Callback execution.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_managers():
  sem = MagicMock(spec=SemanticsManager)
  # Ensure get_framework_config returns safely for hyrdation logic
  sem.get_framework_config.return_value = {}

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return sem, cfg


def test_context_state_initialization(mock_managers):
  sem, cfg = mock_managers
  ctx = RewriterContext(sem, cfg)

  # Check simple state defaults
  assert isinstance(ctx.scope_stack, list)
  assert len(ctx.scope_stack) == 1
  assert isinstance(ctx.alias_map, dict)
  assert ctx.in_module_class is False

  # Check framework props
  assert ctx.source_fw == "torch"
  assert ctx.target_fw == "jax"


def test_hydration_of_source_aliases(mock_managers):
  """Verify alias map pre-population from Config."""
  sem, cfg = mock_managers

  # Mock return for torch config
  sem.get_framework_config.side_effect = lambda fw: ({"alias": {"name": "custom_torch"}} if fw == "torch" else {})

  ctx = RewriterContext(sem, cfg)

  assert "custom_torch" in ctx.alias_map
  assert ctx.alias_map["custom_torch"] == "custom_torch"


def test_callback_dispatch(mock_managers):
  sem, cfg = mock_managers

  mock_injector = MagicMock()
  ctx = RewriterContext(sem, cfg, preamble_injector=mock_injector)

  ctx.hook_context.inject_preamble("print('test')")
  mock_injector.assert_called_once_with("print('test')")


def test_error_accumulation(mock_managers):
  sem, cfg = mock_managers
  ctx = RewriterContext(sem, cfg)

  ctx.current_stmt_errors.append("Fatal Error")
  assert len(ctx.current_stmt_errors) == 1
