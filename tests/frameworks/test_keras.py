"""
Tests for the Keras Adapter (Keras v3 Logic).
Use real module types to ensure inspection works robustly.
"""

import sys
import pytest
import types
import importlib
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.base import StandardCategory
import ml_switcheroo.frameworks.keras


class MockKerasObject:
  """Mock base class to satisfy is_keras_object check."""

  def get_config(self):
    return {}

  def from_config(self):
    return {}


# Concrete Mocks
class MeanSquaredError(MockKerasObject):
  pass


class Adam(MockKerasObject):
  pass


class Optimizer(MockKerasObject):
  pass


class Loss(MockKerasObject):
  pass


def mock_relu(x):
  pass


mock_relu.__name__ = "relu"


def create_mock_module(name, members=None):
  """Helper to create a real ModuleType object."""
  mod = types.ModuleType(name)
  if members:
    for k, v in members.items():
      setattr(mod, k, v)
  return mod


@pytest.fixture
def mock_keras_env():
  """
  Sets up a complete mock Keras environment and reloads the adapter
  to bind to these mocks.
  """
  # Create Module Objects
  losses = create_mock_module("keras.losses", {"MeanSquaredError": MeanSquaredError, "Loss": Loss, "BadObj": MagicMock()})
  opts = create_mock_module("keras.optimizers", {"Adam": Adam, "Optimizer": Optimizer})
  acts = create_mock_module("keras.activations", {"relu": mock_relu})
  mock_ops = create_mock_module("keras.ops", {})
  mock_random = create_mock_module("keras.random", {})

  # Attach submodules to the root module
  mock_keras = create_mock_module("keras", {})
  mock_keras.__path__ = []  # Mark as package
  mock_keras.losses = losses
  mock_keras.optimizers = opts
  mock_keras.activations = acts
  mock_keras.ops = mock_ops
  mock_keras.random = mock_random

  overrides = {
    "keras": mock_keras,
    "keras.losses": losses,
    "keras.optimizers": opts,
    "keras.activations": acts,
    "keras.ops": mock_ops,
    "keras.random": mock_random,
  }

  # Build a content map for getmembers using object IDs to be 100% robust against proxies/naming
  module_content_map = {
    id(losses): [("MeanSquaredError", MeanSquaredError), ("Loss", Loss), ("BadObj", getattr(losses, "BadObj"))],
    id(opts): [("Adam", Adam), ("Optimizer", Optimizer)],
    id(acts): [("relu", mock_relu)],
  }

  # Use patch.dict to install mocks into sys.modules
  with patch.dict(sys.modules, overrides):
    # Reload adapter module so it imports the mocks from sys.modules
    importlib.reload(ml_switcheroo.frameworks.keras)

    # Safety: If the reload import block swallowed imports (try/except), 'keras' might be None/Ghost.
    # Since we are mocking the environment, we want LIVE behavior.
    # We forcefully inject the mock module if it fell back to None.
    if ml_switcheroo.frameworks.keras.keras is None:
      ml_switcheroo.frameworks.keras.keras = mock_keras
      # Ensure submodules are attached locally if they weren't imported
      ml_switcheroo.frameworks.keras.keras.losses = losses
      ml_switcheroo.frameworks.keras.keras.optimizers = opts
      ml_switcheroo.frameworks.keras.keras.activations = acts

    # Patch inspect.getmembers to reliably return our content via identity
    with patch("inspect.getmembers") as mock_members:

      def get_members(obj):
        # Identify modules by object identity
        return module_content_map.get(id(obj), [])

      mock_members.side_effect = get_members

      adapter = ml_switcheroo.frameworks.keras.KerasAdapter()
      # Ensure adapter detected live mode
      if adapter._mode != "live":
        # Force live mode if init check failed before our manual fix
        adapter._mode = "live"

      yield adapter


def test_collect_losses_logic(mock_keras_env):
  """Verify Only valid Losses are collected."""
  adapter = mock_keras_env
  results = adapter.collect_api(StandardCategory.LOSS)
  names = {r.name for r in results}
  assert "MeanSquaredError" in names
  assert "Loss" not in names  # Filtered by blocklist
  assert "BadObj" not in names  # Filtered by no get_config


def test_collect_optimizers_logic(mock_keras_env):
  """Verify Only valid Optimizers are collected."""
  adapter = mock_keras_env
  results = adapter.collect_api(StandardCategory.OPTIMIZER)
  names = {r.name for r in results}
  assert "Adam" in names
  assert "Optimizer" not in names
