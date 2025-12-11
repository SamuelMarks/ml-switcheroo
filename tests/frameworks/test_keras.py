"""
Tests for the Keras Adapter.

Verifies:
1. `collect_api` scanning logic for Losses, Optimizers, Activations.
2. Serialization check heuristics (`get_config`) without conflicting with Naming.
3. Fallback to Ghost Mode when Keras is missing.
"""

import sys
import pytest
import types
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo.frameworks.base import StandardCategory, InitMode

# --- Mocks ---


class MockKerasObject:
  """Simulates a Keras class with serialization methods."""

  def get_config(self):
    return {}


# Rename classes to match actual Keras names so __name__ matches what GhostInspector sets
class MeanSquaredError(MockKerasObject):
  pass


class Adam(MockKerasObject):
  pass


class Optimizer(MockKerasObject):
  pass


def mock_relu(x):
  pass


mock_relu.__name__ = "relu"


# Helper to build a module-like object
def create_mock_module(name, members):
  mod = types.ModuleType(name)
  for k, v in members.items():
    setattr(mod, k, v)
  return mod


@pytest.fixture
def mock_keras_env():
  """Injects a fake 'keras' package into sys.modules."""

  # 1. Losses
  # Matches scan logic (search for class objects in module)
  losses = create_mock_module(
    "keras.losses",
    {
      "MeanSquaredError": MeanSquaredError,  # Valid
      "Loss": MockKerasObject,  # blocked
      "BadObj": MagicMock(),  # No get_config -> skipped
      "_Private": MockKerasObject,  # internal -> skipped
    },
  )

  # 2. Optimizers
  opts = create_mock_module(
    "keras.optimizers",
    {
      "Adam": Adam,  # Valid
      "Optimizer": Optimizer,  # blocked
    },
  )

  # 3. Activations
  acts = create_mock_module("keras.activations", {"relu": mock_relu, "InternalClass": MockKerasObject})

  keras_root = types.ModuleType("keras")
  keras_root.losses = losses
  keras_root.optimizers = opts
  keras_root.activations = acts

  overrides = {"keras": keras_root, "keras.losses": losses, "keras.optimizers": opts, "keras.activations": acts}

  with patch.dict(sys.modules, overrides):
    import importlib
    import ml_switcheroo.frameworks.keras

    importlib.reload(ml_switcheroo.frameworks.keras)

    yield ml_switcheroo.frameworks.keras.KerasAdapter()


def test_collect_losses_logic(mock_keras_env):
  """Verify LOSS scanning uses get_config and ignores base class."""
  adapter = mock_keras_env

  results = adapter.collect_api(StandardCategory.LOSS)

  names = {r.name for r in results}

  assert "MeanSquaredError" in names
  assert "Loss" not in names
  assert "BadObj" not in names
  assert "_Private" not in names


def test_collect_optimizers_logic(mock_keras_env):
  """Verify OPTIMIZER scanning finds Adam and blocks Optimizer."""
  adapter = mock_keras_env

  results = adapter.collect_api(StandardCategory.OPTIMIZER)
  names = {r.name for r in results}

  assert "Adam" in names
  assert "Optimizer" not in names


def test_collect_activations_logic(mock_keras_env):
  """Verify ACTIVATION scanning (Function mode)."""
  adapter = mock_keras_env

  results = adapter.collect_api(StandardCategory.ACTIVATION)

  names = {r.name for r in results}
  assert "relu" in names
  assert "InternalClass" not in names


def test_ghost_mode_fallback_keras():
  """
  Scenario: Keras not installed.
  Expectation: Fallback to Ghost Mode.
  """
  with patch.dict(sys.modules, {"keras": None}):
    import importlib
    import ml_switcheroo.frameworks.keras

    importlib.reload(ml_switcheroo.frameworks.keras)

    dummy_snap = {"version": "3.0", "categories": {}}

    with patch("ml_switcheroo.frameworks.keras.load_snapshot_for_adapter", return_value=dummy_snap):
      adapter = ml_switcheroo.frameworks.keras.KerasAdapter()
      assert adapter._mode == InitMode.GHOST
