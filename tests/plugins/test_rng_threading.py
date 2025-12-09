import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.rng_threading import inject_prng_threading


@pytest.fixture
def rewriter():
  hooks._HOOKS["inject_prng"] = inject_prng_threading
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  op_def = {"variants": {"jax": {"requires_plugin": "inject_prng"}}}

  mgr.get_definition.return_value = ("dropout", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


def test_rng_basic_injection(rewriter):
  code = "def f(x):\n  return torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  assert "def f(rng, x):" in res or "def f(x, rng):" in res
  assert "random.split" in res
  assert "key=" in res


def test_rng_custom_configuration(rewriter):
  rewriter.ctx._runtime_config.plugin_settings = {"rng_arg_name": "seed", "key_var_name": "k"}
  code = "def f(x):\n  torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  assert "def f(seed, x):" in res or "def f(x, seed):" in res
  assert "k = jax" in res


def test_rng_deduplication(rewriter):
  code = "def f(x):\n  torch.dropout(x)\n  torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  # expect 1 split line
  assert res.count("split(rng)") == 1


def test_rng_existing_argument_preserved(rewriter):
  code = "def f(x, rng):\n  torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  # Should not duplicate arg
  def_line = [l for l in res.split("\n") if "def f" in l][0]
  assert def_line.count("rng") == 1
  # Still injects body
  assert "split(rng)" in res
