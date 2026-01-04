"""
Integration test for Type Inference-driven rewriting.
"""

import pytest
import importlib
import ml_switcheroo.core.hooks as hooks
import ml_switcheroo.plugins.rng_threading
import ml_switcheroo.plugins.reshape
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE = """ 
import torch

def process(): 
    x = torch.randn(10) 
    # Implicit method call on inferred Tensor
    y = x.view(5, 2) 
    return y
"""


@pytest.fixture(autouse=True)
def reload_required_plugins():
  """
  Reloads critical plugins to ensure hooks are registered in the global registry.
  This prevents "Missing required plugin" errors caused by test suite isolation/clearing.
  """
  importlib.reload(ml_switcheroo.plugins.rng_threading)
  importlib.reload(ml_switcheroo.plugins.reshape)
  hooks._PLUGINS_LOADED = True


@pytest.fixture
def semantics():
  mgr = SemanticsManager()
  # Ensure definitions loaded
  # torch.randn -> returns Tensor
  # Reshape -> view mapping
  return mgr


def test_inferred_view_rewrite(semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)

  # Pre-check: Ensure semantics have necessary data
  # "torch.randn" should exist. "Reshape" should exist.

  result = engine.run(SOURCE)
  assert result.success

  # Rewriter should map x.view -> jnp.reshape because x is inferred as Tensor
  # This relies on the analyzer correctly tagging 'x'.
  assert "jnp.reshape(x" in result.code
