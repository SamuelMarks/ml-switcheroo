import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks


def rewrite_code(rewriter, code):
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  onehot_def = {
    "std_args": ["input", "num_classes"],
    "variants": {
      "torch": {
        "api": "torch.nn.functional.one_hot",
        "args": {"input": "tensor"},
      },
      "jax": {"api": "jax.nn.one_hot", "args": {"tensor": "x", "input": "x"}},
    },
  }
  mgr.get_definition.side_effect = lambda n: ("OneHot", onehot_def) if "one_hot" in n else None
  mgr.resolve_variant.side_effect = lambda a, f: onehot_def["variants"]["jax"] if a == "OneHot" and f == "jax" else None
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"OneHot": onehot_def}

  # FIX: Populate framework aliases for dynamic module detection
  mgr.framework_configs = {
    "torch": {"alias": {"module": "torch.nn.functional", "name": "F"}},
    "jax": {},
  }

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_onehot_positional(rewriter):
  code = "import torch.nn.functional as F\ny = F.one_hot(x, 10)"
  res = rewrite_code(rewriter, code)

  assert "jax.nn.one_hot" in res
  # Argument re-ordering or normalization happens here
  assert "(x,10)" in res.replace(" ", "")


def test_onehot_kwargs(rewriter):
  code = "import torch.nn.functional as F\ny = F.one_hot(tensor=x, num_classes=5)"
  res = rewrite_code(rewriter, code)
  assert "jax.nn.one_hot" in res
  # Ensure x=x mapping
  assert "x=x" in res
  assert "tensor" not in res
  assert "num_classes=5" in res
