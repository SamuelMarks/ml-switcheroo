"""
Tests for the PyTorch Adapter (Ghost Protocol Compliance).
Updated to verify Distributed Semantics definitions.

Verifies:
1. Dynamic Discovery via mocked torch environment.
2. Ghost Mode fallback when torch is missing.
3. Correct linkage of definitions and imports.
4. Completeness of static definitions (Math, Neural, Optim).
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
        pass

    # Now verify the module state (globals should be None)
    assert ml_switcheroo.frameworks.torch.torch is None

    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value=dummy_snap):
      adapter = TorchAdapter()
      assert adapter._mode == InitMode.GHOST
      results = adapter.collect_api(StandardCategory.LOSS)
      assert results[0].name == "GhostLoss"


def test_definitions_completeness():
  """
  Verify static definitions have been migrated from the Hub.
  Checks key operations for each tier.
  """
  adapter = TorchAdapter()
  defs = adapter.definitions

  # 1. Math
  assert "Abs" in defs
  assert defs["Abs"].api == "torch.abs"
  assert "Einsum" in defs
  assert defs["Einsum"].api == "torch.einsum"

  # 2. Neural Layers
  assert "Linear" in defs
  assert defs["Linear"].api == "torch.nn.Linear"
  assert "Conv2d" in defs
  assert defs["Conv2d"].api == "torch.nn.Conv2d"

  # 3. Activations
  assert "relu" in defs
  assert defs["relu"].api == "torch.nn.functional.relu"

  # 4. Optimization
  assert "Adam" in defs
  assert defs["Adam"].api == "torch.optim.Adam"
  assert "ClipGradNorm" in defs

  # 5. Types & Casting
  assert "Float32" in defs
  assert defs["Float32"].api == "torch.float32"
  assert "CastFloat" in defs
  assert defs["CastFloat"].api == "torch.Tensor.float"

  # 6. Extras / Functional
  assert "vmap" in defs
  assert defs["vmap"].args["in_axes"] == "in_dims"
  assert "no_grad" in defs
  assert defs["no_grad"].api == "torch.no_grad"


def test_imports_exist():
  """Verify import namespaces."""
  adapter = TorchAdapter()
  ns = adapter.import_namespaces
  assert "torch.nn" in ns
  assert "torch.nn.functional" in ns
  # Verify alias suggestions
  assert ns["torch.nn"].recommended_alias == "nn"
  assert ns["torch.nn.functional"].recommended_alias == "F"
