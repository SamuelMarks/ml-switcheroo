"""
Tests for RNG Threading Plugin using Trait-Based Logic.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.rng_threading import inject_prng_threading
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["inject_prng"] = inject_prng_threading
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  # Define op requiring plugin
  op_def = {"variants": {"jax": {"requires_plugin": "inject_prng"}}}
  mgr.get_definition.return_value = ("dropout", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)

  # IMPORTANT: Mock Plugin Traits to enable RNG requirement
  # We use 'jax' as target to pass validation
  mgr.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_explicit_rng=True)}

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_rng_basic_injection(rewriter):
  code = "def f(x):\n  return torch.dropout(x)"
  res = rewrite_code(rewriter, code)

  # Check signature injection
  assert "def f(rng, x):" in res or "def f(x, rng):" in res

  # Check preamble split
  assert "rng, key = jax.random.split(rng)" in res

  # Check call modification
  assert "key=key" in res


def test_rng_custom_configuration(rewriter):
  # Override settings
  rewriter.ctx._runtime_config.plugin_settings = {"rng_arg_name": "seed", "key_var_name": "k"}

  code = "def f(x):\n  torch.dropout(x)"
  res = rewrite_code(rewriter, code)

  # Signature should have 'seed'
  assert "def f(seed, x):" in res or "def f(x, seed):" in res

  # Preamble should split 'seed -> k'
  # Use simpler assertion for string content
  assert "seed, k = jax.random.split(seed)" in res

  # Call should use 'k'
  assert "key=k" in res


def test_no_injection_if_traits_disabled(rewriter):
  # Disable the trait
  rewriter.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_explicit_rng=False)}

  code = "def f(x):\n  return torch.dropout(x)"
  res = rewrite_code(rewriter, code)

  # Should NOT inject anything
  assert "rng" not in res
  assert "split" not in res


def test_rng_deduplication(rewriter):
  code = "def f(x):\n  torch.dropout(x)\n  torch.dropout(x)"
  res = rewrite_code(rewriter, code)
  # Preamble only once
  assert res.count("split(rng)") == 1


def test_remove_generator_arg(rewriter):
  code = "def f(x):\n  torch.dropout(x, generator=g)"
  res = rewrite_code(rewriter, code)
  assert "generator" not in res
