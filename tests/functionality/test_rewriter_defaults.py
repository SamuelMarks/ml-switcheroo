"""
Tests for ODL Default Argument Injection with Rich Types.

Verifies that the `NormalizationMixin` correctly injects default values
defined in the `std_args` of a semantic operation when those arguments
are missing from the source call.

Covers:
- Float injection (e.g., epsilon)
- Boolean injection (e.g., flags)
- List/Container injection (e.g., dims list based logic)
- Preservation of explicit arguments override defaults.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
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

  # 2. Dropout: implicit default None or value
  op_drop = {
    "std_args": [{"name": "x"}, {"name": "p", "type": "float", "default": 0.5}],
    "variants": {
      "torch": {"api": "torch.dropout"},
      "jax": {"api": "jax.random.bernoulli", "args": {"p": "p"}},
    },
  }
  mgr.data["Dropout"] = op_drop
  mgr._reverse_index["torch.dropout"] = ("Dropout", op_drop)

  # 3. Complex: List default
  # Op 'torch.sum(x)' maps to 'j.sum(x, dims=[0,1])' if default injected
  op_complex = {
    "std_args": [{"name": "x"}, {"name": "dims", "default": [0, 1]}],
    "variants": {"torch": {"api": "torch.sum"}, "jax": {"api": "j.sum"}},
  }
  mgr.data["ComplexSum"] = op_complex
  # Must align with the test case code usage: "torch.sum"
  mgr._reverse_index["torch.sum"] = ("ComplexSum", op_complex)

  # 4. Boolean default
  op_bool = {
    "std_args": [{"name": "x"}, {"name": "flag", "default": False}],
    "variants": {"torch": {"api": "torch.flag"}, "jax": {"api": "j.flag"}},
  }
  mgr.data["BoolCheck"] = op_bool
  mgr._reverse_index["torch.flag"] = ("BoolCheck", op_bool)

  # 5. None default
  op_none = {
    "std_args": [{"name": "x"}, {"name": "attr", "default": None}],
    "variants": {"torch": {"api": "torch.none"}, "jax": {"api": "j.none"}},
  }
  mgr.data["NoneCheck"] = op_none
  mgr._reverse_index["torch.none"] = ("NoneCheck", op_none)

  # Mock aliasing for 't' to ensure it's treated as module if needed,
  # though strict test strings imports should handle it.
  mgr.framework_configs["torch"] = {"alias": {"module": "torch", "name": "t"}}
  mgr.framework_configs["jax"] = {}

  return mgr


@pytest.fixture
def rewriter(manager):
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(manager, config)


def rewrite(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


def test_inject_default_float(rewriter):
  """
  Scenario: 'eps' defaults to 1e-5. Input skips it.
  Expectation: Target call contains 'epsilon=1e-05'.
  """
  code = "import torch\ny = torch.nn.LayerNorm(x)"
  res = rewrite(rewriter, code)
  # Check for valid float repr
  assert "epsilon=1e-05" in res or "epsilon=0.00001" in res


def test_preserve_explicit_eps(rewriter):
  """
  Scenario: 'eps' specified in source as 0.1.
  Expectation: Target uses 0.1, default 1e-5 ignored.
  """
  code = "import torch\ny = torch.nn.LayerNorm(x, eps=0.1)"
  res = rewrite(rewriter, code)
  assert "epsilon=0.1" in res
  assert "1e-5" not in res
  assert "1e-05" not in res


def test_inject_default_dropout(rewriter):
  """
  Scenario: 'p' defaults to 0.5. Input skips it.
  Expectation: Target contains 'p=0.5'.
  """
  code = "import torch\ny = torch.dropout(x)"
  res = rewrite(rewriter, code)
  assert "p=0.5" in res


def test_explicit_dropout_keyword(rewriter):
  """
  Scenario: 'p' explicit 0.1.
  Expectation: 'p=0.1'.
  """
  code = "import torch\ny = torch.dropout(x, p=0.1)"
  res = rewrite(rewriter, code)
  assert "p=0.1" in res
  assert "p=0.5" not in res


def test_inject_default_list(rewriter):
  """
  Scenario: 'dims' defaults to [0, 1]. Input 'torch.sum(x)' skips it.
  """
  code = "import torch\ny = torch.sum(x)"
  res = rewrite(rewriter, code)
  assert "dims=[0, 1]" in res


def test_inject_default_bool(rewriter):
  """
  Scenario: 'flag' defaults to False.
  """
  code = "import torch\ny = torch.flag(x)"
  res = rewrite(rewriter, code)
  assert "flag=False" in res


def test_inject_default_none(rewriter):
  """
  Scenario: 'attr' defaults to None.
  Expectation: 'attr=None' injected.
  """
  code = "import torch\ny = torch.none(x)"
  res = rewrite(rewriter, code)
  assert "attr=None" in res
