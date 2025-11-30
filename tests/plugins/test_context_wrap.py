"""
Tests for Context Manager Rewriting Plugin.

Verifies that `torch.no_grad()` contexts are correctly transformed into
`contextlib.nullcontext()` shims for JAX compatibility.
"""

import pytest
from unittest.mock import MagicMock
import libcst as cst

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.context_to_function_wrap import transform_context_manager


# Helper to avoid import errors
def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Register Hook & ensure plugins loaded state is set
  hooks._HOOKS["context_to_function_wrap"] = transform_context_manager
  hooks._PLUGINS_LOADED = True

  # 2. Setup Mock Semantics
  mgr = MagicMock()

  # Define the Semantic Entry
  no_grad_def = {
    "requires_plugin": "context_to_function_wrap",
    # std_args is irrelevant for attribute rewriting now
    # because requires_plugin forces skip in leave_Attribute
    "std_args": ["block"],
    "variants": {
      "torch": {"api": "torch.no_grad"},
      "jax": {"api": "contextlib.nullcontext", "requires_plugin": "context_to_function_wrap"},
    },
  }

  # Setup side_effect for robust lookup
  def get_def_side_effect(name):
    if name == "torch.no_grad":
      return "no_grad_op", no_grad_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = {"no_grad_op": no_grad_def}
  mgr.get_op_by_source.side_effect = lambda api: "no_grad_op" if "no_grad" in api else None

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_no_grad_transformation(rewriter):
  """
  Scenario: Standard usage of `with torch.no_grad():`.
  """
  code = """
import torch
def forward(x):
    with torch.no_grad():
        y = x * 2
    return y
"""
  result = rewrite_code(rewriter, code)

  # 1. Check Preamble Injection
  assert "import contextlib" in result

  # 2. Check Context Replacement
  assert "with contextlib.nullcontext():" in result
  assert "torch.no_grad" not in result


def test_no_grad_as_decorator(rewriter):
  """
  Scenario: `torch.no_grad` used as a decorator.
  """
  code = """
import torch
@torch.no_grad()
def eval_step(x):
    return x
"""
  result = rewrite_code(rewriter, code)

  assert "@contextlib.nullcontext()" in result
  assert "torch.no_grad" not in result


def test_argument_cleaning(rewriter):
  """
  Scenario: User passes arguments to `no_grad`.
  Arguments stripped to ensure nullcontext doesn't crash.
  """
  code = """
import torch
def forward(x):
    with torch.no_grad(ignored_arg=True):
        pass
"""
  result = rewrite_code(rewriter, code)

  # Argument 'ignored_arg' should be removed by the plugin
  assert "contextlib.nullcontext():" in result
  assert "ignored_arg" not in result


def test_plugin_not_triggered_for_others(rewriter):
  """Verify other calls are untouched."""
  # Ensure ignore logic works (get_definition returns None by default via side_effect)
  code = "def f(x): return torch.other_op(x)"
  result = rewrite_code(rewriter, code)
  assert "torch.other_op" in result
