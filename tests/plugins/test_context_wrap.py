"""Test suite for the Context Wrap module."""

import pytest
import typing
from unittest.mock import MagicMock
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.context_to_function_wrap import transform_context_manager


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return typing.cast(str, new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")
    return ""


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["context_to_function_wrap"] = transform_context_manager
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  no_grad_def: dict[str, typing.Any] = {
    "requires_plugin": "context_to_function_wrap",
    "std_args": ["block"],
    "variants": {
      "torch": {"api": "torch.no_grad"},
      "jax": {"api": "contextlib.nullcontext", "requires_plugin": "context_to_function_wrap"},
    },
  }
  mgr.get_definition.side_effect = lambda name: ("no_grad_op", no_grad_def) if name == "torch.no_grad" else None
  mgr.get_known_apis.return_value = {"no_grad_op": no_grad_def}

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    if aid == "no_grad_op" and fw == "jax":
      return typing.cast(dict[str, typing.Any], no_grad_def["variants"]["jax"])
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_no_grad_transformation(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of no grad transformation."""
  code: str = "\nimport torch\ndef f():\n    with torch.no_grad():\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "import contextlib" in result
  assert "with contextlib.nullcontext():" in result


def test_no_grad_as_decorator(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of no grad as decorator."""
  code: str = "\nimport torch\n@torch.no_grad()\ndef eval_step(x):\n    return x\n"
  result: str = rewrite_code(rewriter, code)
  assert "@contextlib.nullcontext()" in result


def test_argument_cleaning(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of argument cleaning."""
  code: str = "\nimport torch\ndef forward(x):\n    with torch.no_grad(ignored_arg=True):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "contextlib.nullcontext():" in result
  assert "ignored_arg" not in result


def test_plugin_not_triggered_for_others(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of plugin not triggered for others."""
  code: str = "def f(x): return torch.other_op(x)"
  result: str = rewrite_code(rewriter, code)
  assert "torch.other_op" in result
