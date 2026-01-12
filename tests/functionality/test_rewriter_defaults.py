"""
Tests for ODL Default Argument Injection with Rich Types.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


@pytest.fixture
def manager():
  """Mock Semantics Manager with ODL definitions containing rich defaults."""
  mgr = SemanticsManager()
  mgr.data = {}
  mgr._reverse_index = {}
  mgr._key_origins = {}
  mgr.framework_configs = {}
  if not hasattr(mgr, "import_data"):
    mgr.import_data = {}

  # 1. LayerNorm: float default (native float type in python)
  op = {
    "std_args": [{"name": "x"}, {"name": "eps", "type": "float", "default": 1e-5}],
    "variants": {
      "torch": {"api": "torch.nn.LayerNorm", "args": {"eps": "eps"}},
      "jax": {"api": "jax.nn.layer_norm", "args": {"eps": "epsilon"}},
    },
  }

  mgr.data["LayerNorm"] = op
  mgr._reverse_index["torch.nn.LayerNorm"] = ("LayerNorm", op)

  # 2. Dropout
  op_drop = {
    "std_args": [{"name": "x"}, {"name": "p", "type": "float", "default": 0.5}],
    "variants": {
      "torch": {"api": "torch.dropout"},
      "jax": {"api": "jax.random.bernoulli", "args": {"p": "p"}},
    },
  }
  mgr.data["Dropout"] = op_drop
  mgr._reverse_index["torch.dropout"] = ("Dropout", op_drop)

  # Mock aliasing
  mgr.framework_configs["torch"] = {"alias": {"module": "torch", "name": "t"}}
  mgr.framework_configs["jax"] = {}

  return mgr


@pytest.fixture
def rewriter(manager):
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(manager, config)


def rewrite(rewriter, code):
  return rewriter.convert(cst.parse_module(code)).code


def test_inject_default_float(rewriter):
  code = "import torch\ny = torch.nn.LayerNorm(x)"
  res = rewrite(rewriter, code)
  # Check for valid float repr
  assert "epsilon=1e-05" in res or "epsilon=0.00001" in res


def test_preserve_explicit_eps(rewriter):
  code = "import torch\ny = torch.nn.LayerNorm(x, eps=0.1)"
  res = rewrite(rewriter, code)
  assert "epsilon=0.1" in res
  assert "1e-5" not in res


def test_inject_default_dropout(rewriter):
  code = "import torch\ny = torch.dropout(x)"
  res = rewrite(rewriter, code)
  assert "p=0.5" in res
