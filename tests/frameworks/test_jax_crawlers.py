"""
Tests for JAX Ecosystem Crawlers (Level 0/1/2).

Verifies that:
1. OptaxScanner correctly identifies loss functions.
2. JAX core adapter finds basic activations.
3. FlaxNNXAdapter delegates to shims and scans correctly.
"""

import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo.frameworks import optax_shim
from ml_switcheroo.frameworks.optax_shim import OptaxScanner
from ml_switcheroo.frameworks.base import StandardCategory
from ml_switcheroo.core.ghost import GhostRef


@pytest.fixture
def mock_optax():
  """
  Mocks optax library structure.
  Provides dummy functions that satisfy inspection checks.
  """
  m_optax = MagicMock()

  # Create dummy functions that pass inspect.isfunction()
  def dummy_loss():
    pass

  dummy_loss.__name__ = "l2_loss"

  def dummy_opt():
    pass

  dummy_opt.__name__ = "adam"

  # Setup structure
  m_optax.losses = MagicMock()
  m_optax.losses.l2_loss = dummy_loss
  m_optax.l2_loss = dummy_loss

  # Set optimizer on root
  m_optax.adam = dummy_opt

  return m_optax


def test_optax_scanner_losses(mock_optax):
  """Verify OptaxScanner finds losses."""
  # We patch optax inside the shim module
  with patch("ml_switcheroo.frameworks.optax_shim.optax", mock_optax):
    # Force getmembers to return our dummy function
    # (MagicMock dir() behavior is unreliable for inspect.getmembers)
    with patch("inspect.getmembers", return_value=[("l2_loss", mock_optax.losses.l2_loss)]):
      results = OptaxScanner.scan_losses()
      names = [r.name for r in results]
      assert "l2_loss" in names


def test_optax_scanner_optimizers(mock_optax):
  """Verify OptaxScanner finds optimizers."""
  with patch("ml_switcheroo.frameworks.optax_shim.optax", mock_optax):
    with patch("inspect.getmembers", return_value=[("adam", mock_optax.adam)]):
      results = OptaxScanner.scan_optimizers()
      names = [r.name for r in results]
      assert "adam" in names


def test_jax_adapter_integration():
  """
  Verify FlaxNNXAdapter calls shims correctly and finds layers.

  We simulate the adapter finding 'l2_loss' via core logic and 'Dense'
  via Flax-specific logic.
  """

  layer_ref = GhostRef(name="Dense", api_path="flax.nnx.Dense", kind="class", params=[])
  loss_ref = GhostRef(name="l2_loss", api_path="optax.l2_loss", kind="func", params=[])

  with patch("ml_switcheroo.frameworks.jax.JaxCoreAdapter.collect_api", return_value=[loss_ref]):
    with patch("ml_switcheroo.frameworks.flax_nnx.FlaxNNXAdapter._scan_nnx_layers", return_value=[layer_ref]):
      # Since we patched the methods, we don't need real imports
      # We just need to ensure __init__ doesn't crash on imports
      with patch.dict(sys.modules, {"flax.nnx": MagicMock()}):
        adapter = FlaxNNXAdapter()
        # force LIVE mode logic
        adapter._mode = "live"

        # Test LOSS delegating to Core
        losses = adapter.collect_api(StandardCategory.LOSS)
        assert losses[0].name == "l2_loss"

        # Test LAYER using Scan Logic
        layers = adapter.collect_api(StandardCategory.LAYER)
        assert layers[0].name == "Dense"
