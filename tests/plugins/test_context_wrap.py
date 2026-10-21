"""Test suite for the Context Wrap module."""

from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.context_to_function_wrap import transform_context_manager
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return str(new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "context_to_function_wrap", getattr({"v": transform_context_manager}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  no_grad_def = {
    "requires_plugin": "context_to_function_wrap",
    "std_args": ["block"],
    "variants": {
      "torch": {"api": "torch.no_grad"},
      "jax": {"api": "contextlib.nullcontext", "requires_plugin": "context_to_function_wrap"},
    },
  }
  mgr.get_definition.side_effect = lambda name: ("no_grad_op", no_grad_def) if name == "torch.no_grad" else None
  mgr.get_known_apis.return_value = {"no_grad_op": no_grad_def}

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "no_grad_op" and fw == "jax":
      res = getattr(
        {"v": getattr({"v": getattr({"v": no_grad_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]},
        "__getitem__",
      )("v")
      return res if isinstance(res, dict) else {}
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_no_grad_transformation(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of no grad transformation.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "\nimport torch\ndef f():\n    with torch.no_grad():\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "import contextlib" in result
  assert "with contextlib.nullcontext():" in result


def test_no_grad_as_decorator(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of no grad as decorator.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "\nimport torch\n@torch.no_grad()\ndef eval_step(x):\n    return x\n"
  result: str = rewrite_code(rewriter, code)
  assert "@contextlib.nullcontext()" in result


def test_argument_cleaning(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of argument cleaning.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "\nimport torch\ndef forward(x):\n    with torch.no_grad(ignored_arg=True):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "contextlib.nullcontext():" in result
  assert "ignored_arg" not in result


def test_plugin_not_triggered_for_others(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of plugin not triggered for others.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "def f(x): return torch.other_op(x)"
  result: str = rewrite_code(rewriter, code)
  assert "torch.other_op" in result
