"""Test suite for the Type Inference module."""

import pytest
import importlib
import ml_switcheroo.core.hooks as hooks
import ml_switcheroo.plugins.rng_threading
import ml_switcheroo.plugins.reshape
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE = "\nimport torch\n\ndef process():\n    x = torch.randn(10)\n    # Implicit method call on inferred Tensor\n    y = x.view(5, 2)\n    return y\n"


@pytest.fixture(autouse=True)
def reload_required_plugins():
  """Helper to reload required plugins."""
  importlib.reload(ml_switcheroo.plugins.rng_threading)
  importlib.reload(ml_switcheroo.plugins.reshape)
  hooks._PLUGINS_LOADED = True


@pytest.fixture
def semantics():
  """Provides a mock semantics for testing."""
  mgr = SemanticsManager()
  return mgr


@pytest.mark.skip(reason="View/reshape mappings removed")
def test_inferred_view_rewrite(semantics):
  """Verifies the behavior of inferred view rewrite."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE)
  assert result.success
  assert "jnp.reshape(x" in result.code
