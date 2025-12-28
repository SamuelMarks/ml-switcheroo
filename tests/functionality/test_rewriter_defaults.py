"""
Tests for ODL Default Argument Injection.

Verifies that if a source call omits an argument, but the ODL specification
explicitly defines a default value, that default is injected into the
target call ensuring semantic correctness across frameworks with different defaults.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OperationDef, ParameterDef, FrameworkVariant


@pytest.fixture
def manager():
  """Mock Semantics Manager with ODL definitions containing explicit defaults."""
  mgr = SemanticsManager()
  mgr.data = {}
  mgr._reverse_index = {}
  mgr._key_origins = {}
  mgr.framework_configs = {}
  if not hasattr(mgr, "import_data"):
    mgr.import_data = {}

  op = {
    "std_args": [{"name": "x"}, {"name": "eps", "type": "float", "default": "1e-5"}],
    "variants": {
      "torch": {"api": "torch.nn.LayerNorm", "args": {"eps": "eps"}},
      "jax": {"api": "jax.nn.layer_norm", "args": {"eps": "epsilon"}},
    },
  }

  mgr.data["LayerNorm"] = op
  mgr._reverse_index["torch.nn.LayerNorm"] = ("LayerNorm", op)

  op_drop = {
    "std_args": [{"name": "x"}, {"name": "p", "type": "float", "default": "0.5"}],
    "variants": {"torch": {"api": "torch.dropout"}, "jax": {"api": "jax.random.bernoulli", "args": {"p": "p"}}},
  }
  mgr.data["Dropout"] = op_drop
  mgr._reverse_index["torch.dropout"] = ("Dropout", op_drop)

  return mgr


@pytest.fixture
def rewriter(manager):
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(manager, config)


def rewrite(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


def test_inject_default_eps(rewriter):
  code = "y = torch.nn.LayerNorm(x)"
  res = rewrite(rewriter, code)
  assert "jax.nn.layer_norm" in res
  assert "epsilon=1e-5" in res


def test_preserve_explicit_eps(rewriter):
  code = "y = torch.nn.LayerNorm(x, eps=0.1)"
  res = rewrite(rewriter, code)
  assert "epsilon=0.1" in res
  assert "epsilon=1e-5" not in res


def test_inject_default_dropout(rewriter):
  code = "y = torch.dropout(x)"
  res = rewrite(rewriter, code)
  assert "p=0.5" in res


def test_explicit_dropout_keyword(rewriter):
  code = "y = torch.dropout(x, p=0.1)"
  res = rewrite(rewriter, code)
  assert "p=0.1" in res
  assert "p=0.5" not in res


def test_explicit_dropout_positional(rewriter):
  """
  Scenario: `torch.dropout(x, 0.2)`
  """
  code = "y = torch.dropout(x, 0.2)"
  res = rewrite(rewriter, code)

  # If passed positionally, normalization logic keeps value (as arg 2)
  # Check for presence of "0.2" in the simplified string
  clean_res = res.replace(" ", "")

  assert "0.2" in clean_res
  # Ensure default injection didn't happen (p=0.5)
  assert "0.5" not in res
