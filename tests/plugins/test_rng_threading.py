"""Test suite for the Rng Threading module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.rng_threading import inject_prng_threading
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  """Rewrites code."""
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["inject_prng"] = inject_prng_threading
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  op_def = {"variants": {"jax": {"requires_plugin": "inject_prng"}}}
  mgr.get_definition.return_value = ("dropout", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)
  mgr.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_explicit_rng=True)}
  mgr.framework_configs = {}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_rng_basic_injection(rewriter):
  """Verifies the behavior of rng basic injection."""
  code = "def f(x):\n  return torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  assert "def f(rng, x):" in res or "def f(x, rng):" in res
  assert "rng, key = jax.random.split(rng)" in res
  assert "key=key" in res


def test_rng_custom_configuration(rewriter):
  """Verifies the behavior of rng custom configuration."""
  rewriter.context.config.plugin_settings = {"rng_arg_name": "seed", "key_var_name": "k"}
  code = "def f(x):\n  torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  assert "def f(seed, x):" in res or "def f(x, seed):" in res
  assert "seed, k = jax.random.split(seed)" in res
  assert "key=k" in res


def test_no_injection_if_traits_disabled(rewriter):
  """Verifies the behavior of no injection if traits disabled."""
  rewriter.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_explicit_rng=False)}
  code = "def f(x):\n  return torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  assert "rng" not in res
  assert "split" not in res


def test_rng_deduplication(rewriter):
  """Verifies the behavior of rng deduplication."""
  code = "def f(x):\n  torch.dropout(x)\n  torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  assert res.count("split(rng)") == 1


def test_remove_generator_arg(rewriter):
  """Removes generator argument."""
  code = "def f(x):\n  torch.dropout(x, generator=g)"
  res = rewrite_code(rewriter, code)
  assert "generator" not in res
