"""
Tests for JAX Ecosystem Crawlers (Level 0/1/2).

Verifies that:
1. OptaxScanner correctly identifies loss functions.
2. JAX core adapter finds basic activations.
3. FlaxNNX adapter leverages shims for optimization.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo.frameworks.optax_shim import OptaxScanner
from ml_switcheroo.frameworks.base import StandardCategory
from ml_switcheroo.core.ghost import GhostRef


@pytest.fixture
def mock_optax():
  """Mocks optax library structure."""
  m_optax = MagicMock()
  m_optax.l2_loss = MagicMock()
  m_optax.adam = MagicMock()
  return m_optax


def test_optax_scanner_losses(mock_optax):
  """Verify OptaxScanner finds losses."""
  with patch.dict(sys.modules, {"optax": mock_optax}):
    results = OptaxScanner.scan_losses()
    names = [r.name for r in results]
    assert "l2_loss" in names


def test_optax_scanner_optimizers(mock_optax):
  """Verify OptaxScanner finds optimizers."""
  with patch.dict(sys.modules, {"optax": mock_optax}):
    results = OptaxScanner.scan_optimizers()
    names = [r.name for r in results]
    assert "adam" in names


def test_jax_adapter_integration():
  """
  Verify FlaxNNXAdapter calls shims correctly and finds layers.

  We simulate the adapter finding 'l2_loss' via core logic and 'Dense'
  via Flax-specific logic.
  """

  # We patch the two underlying discovery methods on the class itself
  # 1. JaxCoreAdapter.collect_api (which Flax inherits/calls for losses)
  # 2. FlaxNNXAdapter._scan_nnx_layers (which finds layers)

  layer_ref = GhostRef(name="Dense", api_path="flax.nnx.Dense", kind="class", params=[])
  loss_ref = GhostRef(name="l2_loss", api_path="optax.l2_loss", kind="func", params=[])

  with patch("ml_switcheroo.frameworks.jax.JaxCoreAdapter.collect_api", return_value=[loss_ref]):
    with patch("ml_switcheroo.frameworks.flax_nnx.FlaxNNXAdapter._scan_nnx_layers", return_value=[layer_ref]):
      # Since we patched the methods, we don't need real imports
      # We just need to ensure __init__ doesn't crash on imports
      with patch.dict(sys.modules, {"flax.nnx": MagicMock()}):
        adapter = FlaxNNXAdapter()
        # Init might set _mode=GHOST if import fails, force it to behave like LIVE for logic
        adapter._mode = "live"

        # Test LOSS delegating to Core
        losses = adapter.collect_api(StandardCategory.LOSS)
        assert losses[0].name == "l2_loss"

        # Test LAYER using Scan Logic
        layers = adapter.collect_api(StandardCategory.LAYER)
        assert layers[0].name == "Dense"
