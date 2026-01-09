"""
Tests for AuxiliaryStage Execution.

Verifies that the standalone AuxiliaryStage correctly applies rewriting logic
for Decorators and Control Flow using the shared RewriterContext.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter.control_flow import AuxiliaryStage
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import _HOOKS, clear_hooks


@pytest.fixture
def mock_managers():
  sem = MagicMock(spec=SemanticsManager)
  # Configure get_framework_config for trait retrieval in DecoratorMixin
  sem.get_framework_config.return_value = {}

  # Configure get_definition for Decorator mapping
  jit_def = {"variants": {"jax": {"api": "jax.jit"}, "torch": {"api": "torch.jit.script"}}}
  sem.get_definition.side_effect = lambda n: ("Jit", jit_def) if "jit" in n else None

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return sem, cfg


@pytest.fixture
def aux_stage(mock_managers):
  sem, cfg = mock_managers
  ctx = RewriterContext(sem, cfg)
  return AuxiliaryStage(ctx)


def test_decorator_rewrite(aux_stage):
  """Verify decorator renaming logic."""
  code = """
@torch.jit.script
def f(x): pass
"""
  tree = cst.parse_module(code)
  new_tree = tree.visit(aux_stage)

  assert "@jax.jit" in new_tree.code
  assert "@torch" not in new_tree.code


def test_loop_safety_hook_trigger(aux_stage):
  """Verify control flow hook execution."""

  clear_hooks()

  # Mock hook that transforms loop
  _HOOKS["transform_for_loop"] = lambda node, ctx: cst.FlattenSentinel([])

  code = """
for i in range(10):
    pass
"""
  tree = cst.parse_module(code)
  new_tree = tree.visit(aux_stage)

  # Hook returns empty list -> loop removed
  assert "for" not in new_tree.code

  clear_hooks()
