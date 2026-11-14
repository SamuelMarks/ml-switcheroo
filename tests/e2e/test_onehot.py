"""Test suite for the Onehot module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  mod = cst.parse_module(code)
  return str(rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for onehot tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  onehot_def: dict[str, JsonValue] = {
    "std_args": ["input", "num_classes"],
    "variants": {
      "torch": {"api": "torch.nn.functional.one_hot", "args": {"input": "tensor"}},
      "jax": {"api": "jax.nn.one_hot", "args": {"tensor": "x", "input": "x"}},
    },
  }

  def side_effect_get_def(n: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets definition.

    Args:
        n: Input `n` parameter value (`str`).

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from side effect get def.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = ("OneHot", onehot_def) if "one_hot" in n else None
    return _ret_v

  def side_effect_resolve(a: str, f: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        a: Input `a` parameter value (`str`).
        f: Input `f` parameter value (`str`).

    Returns:
        Resulting `dict[str, JsonValue] | None` from side effect resolve.
    """
    if a == "OneHot" and f == "jax":
      variants = onehot_def["variants"]
      if isinstance(variants, dict):
        jax_var = variants.get("jax")
        if isinstance(jax_var, dict):
          return jax_var
      return {}
    return None

  mgr.get_definition.side_effect = side_effect_get_def
  mgr.resolve_variant.side_effect = side_effect_resolve
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"OneHot": onehot_def}
  mgr.framework_configs = {"torch": {"alias": {"module": "torch.nn.functional", "name": "F"}}, "jax": {}}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_onehot_positional(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.nn.one_hot` for onehot positional.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "import torch.nn.functional as F\ny = F.one_hot(x, 10)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.nn.one_hot" in res
  assert "(x,10)" in res.replace(" ", "")


def test_onehot_kwargs(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.nn.one_hot` for onehot kwargs.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "import torch.nn.functional as F\ny = F.one_hot(tensor=x, num_classes=5)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.nn.one_hot" in res
  assert "x=x" in res
  assert "tensor" not in res
  assert "num_classes=5" in res
