"""
Tests for the PyTorch Adapter (Ghost Protocol Compliance).

Verifies:
1. Dynamic Discovery via mocked torch environment.
2. Ghost Mode fallback when torch is missing.
3. Correct linkage of definitions and imports.
"""

import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch

import ml_switcheroo.frameworks.torch
from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.base import StandardCategory, InitMode

# --- Mock Class Hierarchy ---


class MockModule:
  """Simulates torch.nn.Module."""

  pass


class MockOptimizer:
  """Simulates torch.optim.Optimizer."""

  pass


# Concrete Implementations
class MSELoss(MockModule):
  pass


class CrossEntropyLoss(MockModule):
  pass


class Conv2d(MockModule):
  pass


class ReLU(MockModule):
  pass


class Adam(MockOptimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
    pass


@pytest.fixture
def mock_torch_hierarchy():
  """
  Creates a robust mock of the torch namespace.
  """
  # Create magic mocks that act as packages
  m_torch = MagicMock()
  m_torch.__path__ = []  # Mark as package

  m_nn = MagicMock()
  m_optim = MagicMock()
  m_modules = MagicMock()
  m_activations = MagicMock()
  m_activations.__name__ = "torch.nn.modules.activation"

  # Link hierarchy
  m_torch.nn = m_nn
  m_nn.modules = m_modules
  m_torch.optim = m_optim

  # Bind base classes for inheritance checks
  m_nn.Module = MockModule
  m_optim.Optimizer = MockOptimizer

  # Populate (for direct access via getattr)
  m_nn.MSELoss = MSELoss
  m_nn.CrossEntropyLoss = CrossEntropyLoss
  m_nn.Conv2d = Conv2d
  m_nn.ReLU = ReLU
  m_optim.Adam = Adam

  overrides = {
    "torch": m_torch,
    "torch.nn": m_nn,
    "torch.optim": m_optim,
    "torch.nn.modules": m_modules,
    "torch.nn.modules.activation": m_activations,
  }

  with patch.dict(sys.modules, overrides):
    # Critical: Reload to update 'nn' and 'optim' global vars in the module
    importlib.reload(ml_switcheroo.frameworks.torch)
    yield overrides


def test_collect_losses(mock_torch_hierarchy):
  """Verify LOSS collection filters correctly."""
  members = [("MSELoss", MSELoss), ("CrossEntropyLoss", CrossEntropyLoss), ("ReLU", ReLU)]

  with patch("inspect.getmembers", side_effect=lambda x: members):
    adapter = TorchAdapter()
    results = adapter.collect_api(StandardCategory.LOSS)

  names = {r.name for r in results}
  assert "MSELoss" in names
  assert "CrossEntropyLoss" in names
  assert "ReLU" not in names  # Incorrect suffix


def test_collect_activations(mock_torch_hierarchy):
  """Verify ACTIVATION collection via submodule and fallback."""
  activation_mod = mock_torch_hierarchy["torch.nn.modules.activation"]

  def mock_getmembers(obj):
    # Check if object is the activation module mock
    if obj == activation_mod or getattr(obj, "__name__", "") == "torch.nn.modules.activation":
      return [("ReLU", ReLU)]
    # Fallback to main nn
    return [("MSELoss", MSELoss), ("ReLU", ReLU), ("Conv2d", Conv2d)]

  with patch("inspect.getmembers", side_effect=mock_getmembers):
    adapter = TorchAdapter()
    results = adapter.collect_api(StandardCategory.ACTIVATION)

  names = {r.name for r in results}
  assert "ReLU" in names
  assert "Conv2d" not in names


def test_collect_optimizers(mock_torch_hierarchy):
  """Verify OPTIMIZER collection handles params."""
  members = [("Adam", Adam), ("Optimizer", MockOptimizer)]
  with patch("inspect.getmembers", return_value=members):
    adapter = TorchAdapter()
    results = adapter.collect_api(StandardCategory.OPTIMIZER)

  assert len(results) == 1
  assert results[0].name == "Adam"
  assert results[0].has_arg("lr")


def test_ghost_mode_fallback():
  """Verify graceful degradation when torch is missing."""
  dummy_snap = {
    "version": "1.0",
    "categories": {"loss": [{"name": "GhostLoss", "api_path": "t.GhostLoss", "kind": "class", "params": []}]},
  }

  # Force import failure by clearing dict and setting None
  # We reuse patch.dict logic carefully
  with patch.dict(sys.modules):
    # Ensure import fails
    if "torch" in sys.modules:
      del sys.modules["torch"]
    # We mock 'ml_switcheroo.frameworks.torch' loading behavior?
    # Actually, if we reload, it will try to import 'torch'.
    # We want it to FAIL inside the reload/import.

    # We patch built-in import to fail for 'torch'
    original_import = __import__

    def fail_torch_import(name, *args, **kwargs):
      if name == "torch":
        raise ImportError("No Torch")
      return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_torch_import):
      try:
        importlib.reload(ml_switcheroo.frameworks.torch)
      except ImportError:
        # If reload fails completely (it shouldn't, due to catch block), good.
        # The module catches ImportError and sets variables to None.
        pass

    # Now verify the module state (globals should be None)
    assert ml_switcheroo.frameworks.torch.torch is None

    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value=dummy_snap):
      adapter = TorchAdapter()
      assert adapter._mode == InitMode.GHOST
      results = adapter.collect_api(StandardCategory.LOSS)
      assert results[0].name == "GhostLoss"


def test_definitions_exist():
  """Verify static definitions."""
  adapter = TorchAdapter()
  defs = adapter.definitions
  assert "Linear" in defs
  assert defs["Linear"].api == "torch.nn.Linear"


def test_imports_exist():
  """Verify import namespaces."""
  adapter = TorchAdapter()
  ns = adapter.import_namespaces
  assert "torch.nn" in ns
