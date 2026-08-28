"""Test suite for the Onehot module."""

import pytest
import libcst as cst
import typing
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  mod = cst.parse_module(code)
  return typing.cast(str, rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing."""
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  onehot_def: dict[str, typing.Any] = {
    "std_args": ["input", "num_classes"],
    "variants": {
      "torch": {"api": "torch.nn.functional.one_hot", "args": {"input": "tensor"}},
      "jax": {"api": "jax.nn.one_hot", "args": {"tensor": "x", "input": "x"}},
    },
  }

  def side_effect_get_def(n: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets definition."""
    return ("OneHot", onehot_def) if "one_hot" in n else None

  def side_effect_resolve(a: str, f: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves variant."""
    return typing.cast(dict[str, typing.Any], onehot_def["variants"]["jax"]) if a == "OneHot" and f == "jax" else None

  mgr.get_definition.side_effect = side_effect_get_def
  mgr.resolve_variant.side_effect = side_effect_resolve
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"OneHot": onehot_def}
  mgr.framework_configs = {"torch": {"alias": {"module": "torch.nn.functional", "name": "F"}}, "jax": {}}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_onehot_positional(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of onehot positional."""
  code: str = "import torch.nn.functional as F\ny = F.one_hot(x, 10)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.nn.one_hot" in res
  assert "(x,10)" in res.replace(" ", "")


def test_onehot_kwargs(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of onehot keyword arguments."""
  code: str = "import torch.nn.functional as F\ny = F.one_hot(tensor=x, num_classes=5)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.nn.one_hot" in res
  assert "x=x" in res
  assert "tensor" not in res
  assert "num_classes=5" in res
