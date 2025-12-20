"""
Tests for the PyTorch Adapter (Ghost Protocol Compliance).

This module verifies the integration between the Switcheroo PyTorch Adapter
and the underlying PyTorch library (simulated via mocks). It ensures that
the adapter can dynamically discover API surfaces when PyTorch is installed,
and gracefully degrade to "Ghost Mode" using JSON snapshots when it is not.

Refactor: Also verifies 'definitions' property contains expected keys.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.base import StandardCategory, InitMode

# --- Mock Class Hierarchy ---
# We define these globally so they can be inspected easily by the adapter.
# These replace the fragile 'types.ModuleType' approach with real inheritance.


class MockModule:
  """Simulates torch.nn.Module base class."""

  pass


class MockOptimizer:
  """Simulates torch.optim.Optimizer base class."""

  pass


# 1. Losses (Must inherit MockModule + end with 'Loss')
class MSELoss(MockModule):
  """Standard Regression Loss."""

  def __init__(self, size_average=None, reduce=None, reduction="mean"):
    pass


class CrossEntropyLoss(MockModule):
  """Standard Classification Loss."""

  def __init__(self, weight=None, reduction="mean"):
    pass


# 2. Layers (Nhert MockModule, but NOT activation/loss)
class Conv2d(MockModule):
  """Standard Convoluational Layer."""

  def __init__(self, in_channels, out_channels, kernel_size):
    pass


# 3. Activations (Inherit MockModule, specific submodule or name list)
class ReLU(MockModule):
  """Standard Activation."""

  def __init__(self, inplace=False):
    pass


# 4. Optimizers (Inherit MockOptimizer)
class Adam(MockOptimizer):
  """Standard Optimizer."""

  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
    pass


@pytest.fixture
def mock_torch_hierarchy():
  """
  Creates a patch set that simulates a valid torch environment.

  It constructs the `sys.modules` structure required for the adapter to:
  1. Import `torch`, `torch.nn`, `torch.optim`.
  2. Perform `issubclass` checks against the Mock bases.

  Returns:
      dict: A dictionary suitable for patch.dict(sys.modules, ...).
  """
  # Create module mocks
  m_torch = MagicMock()
  m_nn = MagicMock()
  m_optim = MagicMock()
  m_modules = MagicMock()
  m_activations = MagicMock()
  m_activations.__name__ = "torch.nn.modules.activation"

  # Link hierarchy
  m_torch.nn = m_nn
  m_nn.modules = m_modules
  m_torch.optim = m_optim

  # 1. Structure Definitions (Crucial for inheritance checks inside adapter)
  m_nn.Module = MockModule
  m_optim.Optimizer = MockOptimizer

  # 2. Populate Contents (simulating what inspect.getmembers would see)
  m_nn.MSELoss = MSELoss
  m_nn.CrossEntropyLoss = CrossEntropyLoss
  m_nn.Conv2d = Conv2d
  m_nn.ReLU = ReLU

  m_optim.Adam = Adam

  # 3. Activation Submodule structure (Used by dynamic activation scanner)
  m_activations.ReLU = ReLU
  # Conv2d is explicitly NOT here

  return {
    "torch": m_torch,
    "torch.nn": m_nn,
    "torch.optim": m_optim,
    "torch.nn.modules": m_modules,
    "torch.nn.modules.activation": m_activations,
  }


def test_collect_losses(mock_torch_hierarchy):
  """
  Verify LOSS category collection logic.

  Expectation:
  - Classes ending in 'Loss' inheriting from nn.Module are collected.
  - Other modules (ReLU, Conv2d) are ignored even if they match inheritance.
  - Signatures include specific arguments like `reduction`.
  """
  with patch.dict(sys.modules, mock_torch_hierarchy):
    # Patch the class reference inside the adapter to ensure issubclass works
    # (MockModule must be the exact same object)
    with patch("ml_switcheroo.frameworks.torch.nn.Module", MockModule):
      # Control inspection results
      members = [("MSELoss", MSELoss), ("CrossEntropyLoss", CrossEntropyLoss), ("ReLU", ReLU)]

      with patch("inspect.getmembers", side_effect=lambda x: members):
        adapter = TorchAdapter()
        results = adapter.collect_api(StandardCategory.LOSS)

  names = {r.name for r in results}
  assert "MSELoss" in names
  assert "CrossEntropyLoss" in names

  # ReLU inherits from MockModule but name doesn't end in Loss -> Filtered
  assert "ReLU" not in names

  # Verify Signature Extraction
  mse_ref = next(r for r in results if r.name == "MSELoss")
  assert mse_ref.has_arg("reduction"), "Failed to extract constructor arg 'reduction'"


def test_collect_activations(mock_torch_hierarchy):
  """
  Verify ACTIVATION category collection logic.

  Expectation:
  - The adapter prefers scanning `torch.nn.modules.activation`.
  - It finds `ReLU`.
  - It explicitly filters out `Conv2d` (a layer logic mismatch) or `MSELoss`.
  """
  activation_mod = mock_torch_hierarchy["torch.nn.modules.activation"]

  with patch.dict(sys.modules, mock_torch_hierarchy):
    with patch("ml_switcheroo.frameworks.torch.nn.Module", MockModule):
      # The adapter's _scan_activations tries importing/scanning the submodule first.
      # Our mock_torch_hierarchy provides 'torch.nn.modules.activation' containing only ReLU.

      # We patch inspect.getmembers to behave differently depending on the module scanning
      # The equality check must compare against the specific mock instance.
      def mock_getmembers(obj):
        # If checking the activation submodule
        # Check against name or object
        if obj == activation_mod or getattr(obj, "__name__", "") == "torch.nn.modules.activation":
          return [("ReLU", ReLU)]
        # If checking main nn (fallback logic test)
        return [("MSELoss", MSELoss), ("ReLU", ReLU), ("Conv2d", Conv2d)]

      with patch("inspect.getmembers", side_effect=mock_getmembers):
        adapter = TorchAdapter()
        results = adapter.collect_api(StandardCategory.ACTIVATION)

  names = {r.name for r in results}
  assert "ReLU" in names

  # Verify strict filtering of non-activations
  assert "Conv2d" not in names
  assert "MSELoss" not in names


def test_collect_optimizers(mock_torch_hierarchy):
  """
  Verify OPTIMIZER category collection logic.

  Expectation:
  - Finds classes inheriting from `torch.optim.Optimizer`.
  - Extracts `lr` and `betas` arguments.
  """
  with patch.dict(sys.modules, mock_torch_hierarchy):
    # Patch the Optimizer base class check
    with patch("ml_switcheroo.frameworks.torch.optim.Optimizer", MockOptimizer):
      members = [("Adam", Adam), ("Optimizer", MockOptimizer)]

      with patch("inspect.getmembers", return_value=members):
        adapter = TorchAdapter()
        results = adapter.collect_api(StandardCategory.OPTIMIZER)

  assert len(results) == 1
  assert results[0].name == "Adam"

  # Check signature details
  assert results[0].has_arg("lr")
  assert results[0].has_arg("betas")


def test_ghost_mode_fallback():
  """
  Scenario: Torch is NOT installed.

  Expectation:
  - Adapter initializes without crashing.
  - Adapter enters `InitMode.GHOST` (not LIVE).
  - Adapter uses `load_snapshot_for_adapter` to return cached data.
  """
  # Mock the snapshot loader response
  dummy_snap = {
    "version": "1.0",
    "categories": {"loss": [{"name": "GhostLoss", "api_path": "t.GhostLoss", "kind": "class", "params": []}]},
  }

  # Patch 1: sys.modules to simulate failure of 'import torch' inside methods
  # Patch 2: The module-level 'torch' variable in the framework file, forcing __init__ to see 'None'
  with patch.dict(sys.modules, {"torch": None}):
    with patch("ml_switcheroo.frameworks.torch.torch", None):
      with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value=dummy_snap):
        # Initialize
        adapter = TorchAdapter()

        # Critical Assert: Ensure it detected absence of torch
        assert adapter._mode == InitMode.GHOST

        # Ensure it read from snapshot
        results = adapter.collect_api(StandardCategory.LOSS)
        assert len(results) == 1
        assert results[0].name == "GhostLoss"


def test_definitions_exist():
  """Verify static definitions include migrated keys."""
  adapter = TorchAdapter()
  defs = adapter.definitions
  assert "Linear" in defs
  assert "vmap" in defs
  assert "Adam" in defs
  # Verify content
  assert defs["Linear"].api == "torch.nn.Linear"
  assert defs["vmap"].args["in_axes"] == "in_dims"


def test_imports_exist():
  """Verify import namespaces are defined."""
  adapter = TorchAdapter()
  ns = adapter.import_namespaces
  assert "torch.nn" in ns
  assert "torchvision" in ns
