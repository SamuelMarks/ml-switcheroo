"""
Tests for the PyTorch Adapter (Ghost Protocol Compliance).

Verifies:
1. `collect_api(LOSS)` finds standard Torch losses.
2. `collect_api(OPTIMIZER)` finds standard Torch optimizers.
3. `collect_api(ACTIVATION)` finds standard activations.
4. Signatures (GhostRef) are correctly populated.
5. Logic gracefully handles missing torch installation (simulated).
6. Ghost Mode hydration works when snapshots are present.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.base import StandardCategory, InitMode
from ml_switcheroo.core.ghost import GhostRef

# --- Mocking Torch Structure ---
# We define Real classes so issubclass() works as expected.


class _Module:
  """Mock for torch.nn.Module"""

  pass


class _Loss:
  """Mock base for losses"""

  pass


class _Optimizer:
  """Mock base for optimizers"""

  pass


class MSELoss(_Loss):
  __module__ = "torch.nn.modules.loss"

  def __init__(self, size_average=None, reduce=None, reduction="mean"):
    pass


class CrossEntropyLoss(_Loss):
  __module__ = "torch.nn.modules.loss"

  def __init__(self, weight=None, reduction="mean"):
    pass


class Adam(_Optimizer):
  __module__ = "torch.optim.adam"

  def __init__(self, params, lr=0.001, betas=(0.9, 0.999)):
    pass


class ReLU(_Module):
  __module__ = "torch.nn.modules.activation"

  def __init__(self, inplace=False):
    pass


class Conv2d(_Module):
  """Not an activation"""

  __module__ = "torch.nn.modules.conv"


@pytest.fixture
def mock_torch_env():
  """
  Simulates an environment where 'torch', 'torch.nn', 'torch.optim' are importable.
  Uses types.ModuleType to ensure class attributes are respected.
  """
  # Create module mocks
  mock_torch = types.ModuleType("torch")
  mock_nn = types.ModuleType("torch.nn")
  mock_optim = types.ModuleType("torch.optim")

  # Link hierarchy
  mock_torch.nn = mock_nn
  mock_torch.optim = mock_optim

  # Link module bases for issubclass checks
  mock_nn.Module = _Module
  mock_optim.Optimizer = _Optimizer
  # torch.nn.modules.loss._Loss is internal, usually not exposed directly on nn,
  # but our adapter doesn't strict check _Loss inheritance if it uses name heuristics.

  # Manually populate attributes for inspection logic
  # LOSSES
  mock_nn.MSELoss = MSELoss
  mock_nn.CrossEntropyLoss = CrossEntropyLoss
  # ACTIVATIONS
  mock_nn.ReLU = ReLU
  # OTHER LAYERS
  mock_nn.Conv2d = Conv2d

  # OPTIMIZERS
  mock_optim.Adam = Adam

  # Patch sys.modules
  overrides = {
    "torch": mock_torch,
    "torch.nn": mock_nn,
    "torch.optim": mock_optim,
  }

  with patch.dict(sys.modules, overrides):
    # Force reload to pick up imports from the patched sys.modules
    import importlib
    import ml_switcheroo.frameworks.torch

    importlib.reload(ml_switcheroo.frameworks.torch)

    adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
    yield adapter


def test_collect_losses(mock_torch_env):
  """Verify LOSS category collection logic."""
  adapter = mock_torch_env

  # Mock inspect.getmembers because types.ModuleType iteration
  # yields actual members, which is fine, but patching allows control.
  # Actually, with types.ModuleType populated above, standard inspect.getmembers
  # SHOULD work fine without patching if we populated __dict__.
  # BUT, we set attributes on module instance, so dir() usually finds them.

  # Let's rely on manual patching of inspect.getmembers to be deterministic about what "exists"
  # and avoid noise from builtins.
  members_nn = [
    ("MSELoss", MSELoss),
    ("CrossEntropyLoss", CrossEntropyLoss),
    ("ReLU", ReLU),  # Should be filtered
  ]

  with patch("inspect.getmembers", return_value=members_nn):
    results = adapter.collect_api(StandardCategory.LOSS)

  names = {r.name for r in results}
  assert "MSELoss" in names
  assert "CrossEntropyLoss" in names
  assert "ReLU" not in names

  # Verify signature extraction (GhostRef)
  ref = next(r for r in results if r.name == "MSELoss")
  assert ref.has_arg("reduction")


def test_collect_optimizers(mock_torch_env):
  """Verify OPTIMIZER category collection logic."""
  adapter = mock_torch_env

  # Explicitly list members to return for 'optim' scan
  members_optim = [
    ("Adam", Adam),
    ("Optimizer", _Optimizer),  # Base class, ignored
  ]

  with patch("inspect.getmembers", return_value=members_optim):
    results = adapter.collect_api(StandardCategory.OPTIMIZER)

  assert len(results) == 1
  assert results[0].name == "Adam"

  # Verify signature
  assert results[0].has_arg("lr")
  assert results[0].has_arg("betas")


def test_collect_activations(mock_torch_env):
  """Verify ACTIVATION category collection logic."""
  adapter = mock_torch_env

  members_nn = [
    ("MSELoss", MSELoss),
    ("ReLU", ReLU),
    ("Conv2d", Conv2d),  # Module but not activation
  ]

  with patch("inspect.getmembers", return_value=members_nn):
    results = adapter.collect_api(StandardCategory.ACTIVATION)

  names = {r.name for r in results}
  assert "ReLU" in names
  assert "MSELoss" not in names
  assert "Conv2d" not in names


def test_ghost_mode_fallback():
  """
  Scenario: Torch is NOT installed.
  Expectation: Adapter enters GHOST mode.
  """
  # Force Import Error
  with patch.dict(sys.modules, {"torch": None}):
    import importlib
    import ml_switcheroo.frameworks.torch

    # Reload to trigger the "try: import torch except: torch=None" block
    importlib.reload(ml_switcheroo.frameworks.torch)

    # We patch load_snapshot_for_adapter to return dummy data
    dummy_snap = {
      "version": "1.0",
      "categories": {"loss": [{"name": "GhostLoss", "api_path": "t.GhostLoss", "kind": "class", "params": []}]},
    }

    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value=dummy_snap):
      adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
      assert adapter._mode == InitMode.GHOST

      results = adapter.collect_api(StandardCategory.LOSS)
      assert len(results) == 1
      assert results[0].name == "GhostLoss"
