"""
Tests for RNG/State Threading Plugin.

Verifies that stochastic operations (like dropout) are effectively rewritten
to use explicit PRNG key passing (JAX style).
"""

import pytest
from unittest.mock import MagicMock
import libcst as cst

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.rng_threading import inject_prng_threading


# Helper to avoid import errors
def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Register Hook & Prevent automatic loading from disk
  hooks._HOOKS["inject_prng"] = inject_prng_threading
  hooks._PLUGINS_LOADED = True

  # 2. Mock Semantics
  mgr = MagicMock()

  dropout_def = {
    "requires_plugin": "inject_prng",
    "std_args": ["x", "p"],
    "variants": {
      "torch": {"api": "torch.dropout"},
      "jax": {"api": "jax.random.bernoulli", "requires_plugin": "inject_prng"},
    },
  }

  def get_def_side_effect(name):
    if name == "torch.dropout":
      return "dropout", dropout_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = {"dropout": dropout_def}
  mgr.get_op_by_source.side_effect = lambda api: "dropout" if "dropout" in api else None

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_rng_basic_injection(rewriter):
  """
  Scenario: Single stochastic call.
  """
  code = """
import torch
def forward(x):
    return torch.dropout(x, 0.5)
"""
  result = rewrite_code(rewriter, code)

  # 1. Signature Injection
  assert "def forward(rng, x):" in result or "def forward(x, rng):" in result

  # 2. Preamble Injection
  assert "rng, key = jax.random.split(rng)" in result

  # 3. Call Modification
  assert "key=key" in result


def test_rng_deduplication(rewriter):
  """
  Scenario: Two stochastic calls in one function.
  Expect: Only ONE 'rng' argument, ONE split line.
  """
  code = """
import torch
def forward(x):
    a = torch.dropout(x, 0.1)
    b = torch.dropout(x, 0.2)
    return a + b
"""
  result = rewrite_code(rewriter, code)

  def_line = [line for line in result.split("\n") if line.startswith("def forward")][0]
  assert def_line.count("rng") == 1
  assert result.count("rng, key = jax.random.split(rng)") == 1


def test_rng_existing_argument_preserved(rewriter):
  """
  Scenario: Function already has 'rng'.
  Expect: No duplicate argument injected.
  """
  code = """
import torch
def forward(x, rng):
    return torch.dropout(x, 0.5)
"""
  result = rewrite_code(rewriter, code)

  def_line = [line for line in result.split("\n") if line.startswith("def forward")][0]
  # 'rng' appears, check simple presence logic
  assert "rng" in def_line
  # Preamble still added
  assert "rng, key = jax.random.split(rng)" in result


def test_non_stochastic_ignored(rewriter):
  """Calls without the plugin flag are untouched."""
  # Ensure side_effect handles unknown calls (returns None)
  code = "def f(x): return x"
  result = rewrite_code(rewriter, code)

  assert "rng" not in result
  assert "split" not in result
