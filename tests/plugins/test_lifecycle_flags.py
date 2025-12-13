"""
Tests for Lifecycle Flags Plugin.

Verifies that imperative model state changes are correctly converted
to functional boolean flags.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.lifecycle_flags import convert_lifecycle_flags
from ml_switcheroo.core.rewriter import PivotRewriter


def rewrite_code(rewriter, code: str) -> str:
  """Helper to run the visitor on code string."""
  tree = cst.parse_module(code)
  # We must traverse wrappers because the hook targets `Expr` nodes,
  # and `PivotRewriter` manages traversal.
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Manual Hook Registration for test isolation
  hooks._HOOKS.clear()  # Clear entries
  hooks._HOOKS["convert_lifecycle"] = convert_lifecycle_flags
  hooks._PLUGINS_LOADED = True

  # 2. Mock Dependency Manager
  # The Semantic Manager tells the Rewriter WHICH hook to apply for a given API.
  # However, this plugin hooks `Expr` nodes generically or relies on matching.
  # Since `Expr` isn't an API call in the semantic map, we need the Rewriter
  # to trigger generic node visitors.

  # *Fix*: The generic PivotRewriter usually looks up `Call` nodes.
  # For `Expr` node transformers, we usually need to register them as
  # generic transformers or modify the Rewriter to accept node-type subscriptions.

  # *Test Strategy*: Instead of full integration with PivotRewriter (which is complex),
  # we will implement a lightweight visitor subclass for this test that mimics
  # the behavior of applying the "convert_lifecycle" hook to every Expr node.

  class TestVisitor(cst.CSTTransformer):
    def __init__(self):
      self.context = MagicMock()
      self.context.target_fw = "jax"  # Target JAX to trigger rewrite

    def leave_Expr(self, original, updated):
      # Apply our specific hook
      return convert_lifecycle_flags(updated, self.context)

  return TestVisitor()


def test_convert_train_mode(rewriter):
  """
  Scenario: User writes `model.train()`.
  Expectation: Converted to `training = True`.
  """
  code = "model.train()"
  res = rewrite_code(rewriter, code)
  assert "training = True" in res
  assert "model.train()" not in res


def test_convert_eval_mode(rewriter):
  """
  Scenario: User writes `model.eval()`.
  Expectation: Converted to `training = False`.
  """
  code = "model.eval()"
  res = rewrite_code(rewriter, code)
  assert "training = False" in res
  assert "model.eval()" not in res


def test_convert_train_false(rewriter):
  """
  Scenario: User writes `model.train(False)`.
  Expectation: Converted to `training = False`.
  """
  code = "model.train(False)"
  res = rewrite_code(rewriter, code)
  assert "training = False" in res


def test_target_pass_through_torch():
  """
  Scenario: Target is Torch.
  Expectation: Code remains `model.train()`.
  """
  ctx = MagicMock()
  ctx.target_fw = "torch"

  # Manually invoke hook on a node
  node = cst.parse_statement("model.train()")  # Returns SimpleStatementLine containing Expr
  # We need the inner Expr
  expr_node = node.body[0]

  res = convert_lifecycle_flags(expr_node, ctx)

  # Should perform identify transformation (return same node)
  assert res == expr_node or res.value.func.attr.value == "train"


def test_ignore_non_lifecycle_calls(rewriter):
  """
  Scenario: User writes `model.forward()`.
  Expectation: Untouched.
  """
  code = "model.forward()"
  res = rewrite_code(rewriter, code)
  assert "model.forward()" in res
  assert "training =" not in res


def test_ignore_nested_calls(rewriter):
  """
  Scenario: Call is not a standalone statement: `print(model.train())`.
  Expectation: Untouched (replacing with assignment would be invalid).
  """
  code = "print(model.train())"
  res = rewrite_code(rewriter, code)
  # The hook only targets `Expr` (statements), so nested calls inside `print`
  # (which is the Expr) are not Attribute calls at the root.
  # The visitor visits the `print(...)` Expr.
  # The value is a Call to `print`, not `train`.
  # `model.train()` is an arg. It should be ignored safely.
  assert "print(model.train())" in res
