"""
Tests for the Keras Adapter (Keras v3 Logic).
"""

import sys
import pytest
import types
import importlib
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.base import StandardCategory
import ml_switcheroo.frameworks.keras


class MockKerasObject:
  def get_config(self):
    return {}

  def from_config(self):
    return {}


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


# This helps GhostInspector identify name
mock_relu.__name__ = "relu"


# Note: argument signature change to accept dict immediately for cleaner setup
def create_mock_module(name, members=None):
  mod = types.ModuleType(name)
  if members:
    for k, v in members.items():
      setattr(mod, k, v)
  return mod


@pytest.fixture
def mock_keras_env():
  losses = create_mock_module(
    "keras.losses",
    {
      "MeanSquaredError": MeanSquaredError,
      "Loss": Loss,
      "BadObj": MagicMock(),
    },
  )

  opts = create_mock_module(
    "keras.optimizers",
    {
      "Adam": Adam,
      "Optimizer": Optimizer,
    },
  )

  acts = create_mock_module("keras.activations", {"relu": mock_relu})
  mock_ops = create_mock_module("keras.ops", {})
  mock_random = create_mock_module("keras.random", {})

  # FIX: Attach submodules to the root module so `keras.losses` usage works
  mock_keras = create_mock_module("keras", {})
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

  with patch.dict(sys.modules, overrides):
    importlib.reload(ml_switcheroo.frameworks.keras)

    with patch("inspect.getmembers") as mock_members:

      def get_members(obj):
        if obj == losses:
          return [("MeanSquaredError", MeanSquaredError), ("Loss", Loss), ("BadObj", getattr(losses, "BadObj"))]
        if obj == opts:
          return [("Adam", Adam), ("Optimizer", Optimizer)]
        if obj == acts:
          return [("relu", mock_relu)]
        return []

      mock_members.side_effect = get_members
      yield ml_switcheroo.frameworks.keras.KerasAdapter()


def test_collect_losses_logic(mock_keras_env):
  adapter = mock_keras_env
  results = adapter.collect_api(StandardCategory.LOSS)
  names = {r.name for r in results}
  assert "MeanSquaredError" in names
  assert "Loss" not in names
  assert "BadObj" not in names


def test_collect_optimizers_logic(mock_keras_env):
  adapter = mock_keras_env
  results = adapter.collect_api(StandardCategory.OPTIMIZER)
  names = {r.name for r in results}
  assert "Adam" in names
  assert "Optimizer" not in names
