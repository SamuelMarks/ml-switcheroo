"""Test suite for the Onehot module."""

from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

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
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  onehot_def = {
    "std_args": ["input", "num_classes"],
    "variants": {
      "torch": {"api": "torch.nn.functional.one_hot", "args": {"input": "tensor"}},
      "jax": {"api": "jax.nn.one_hot", "args": {"tensor": "x", "input": "x"}},
    },
  }

  def side_effect_get_def(n: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets definition.

    Args:
        n: The n (str) for side effect get def.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from side effect get def.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": (getattr({"v": "OneHot"}, "__getitem__")("v"), getattr({"v": onehot_def}, "__getitem__")("v"))
        if getattr({"v": "one_hot"}, "__getitem__")("v") in getattr({"v": n}, "__getitem__")("v")
        else None
      },
      "__getitem__",
    )("v")
    return _ret_v

  def side_effect_resolve(a: str, f: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        a: The a (str) for side effect resolve.
        f: The f (str) for side effect resolve.

    Returns:
        Resulting `dict[str, JsonValue] | None` from side effect resolve.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": (
          getattr({"v": getattr({"v": onehot_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]
          if isinstance(
            getattr(
              {
                "v": getattr({"v": getattr({"v": onehot_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]
              },
              "__getitem__",
            )("v"),
            dict,
          )
          else {}
        )
        if a == "OneHot" and f == "jax"
        else None
      },
      "__getitem__",
    )("v")
    return _ret_v

  mgr.get_definition.side_effect = side_effect_get_def
  mgr.resolve_variant.side_effect = side_effect_resolve
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"OneHot": onehot_def}
  mgr.framework_configs = {"torch": {"alias": {"module": "torch.nn.functional", "name": "F"}}, "jax": {}}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_onehot_positional(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of onehot positional.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "import torch.nn.functional as F\ny = F.one_hot(x, 10)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.nn.one_hot" in res
  assert "(x,10)" in res.replace(" ", "")


def test_onehot_kwargs(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of onehot keyword arguments.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "import torch.nn.functional as F\ny = F.one_hot(tensor=x, num_classes=5)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.nn.one_hot" in res
  assert "x=x" in res
  assert "tensor" not in res
  assert "num_classes=5" in res
