"""
Integration Tests for Keras Sequential Porting.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential
from ml_switcheroo.core.hooks import _HOOKS

SOURCE_TORCH = """
import torch.nn as nn
def get_model():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    return model
"""


@pytest.fixture
def keras_semantics():
  _HOOKS["keras_sequential_pack"] = transform_keras_sequential

  mgr = MagicMock(spec=SemanticsManager)

  mappings = {
    "Sequential": {
      "std_args": ["layers"],
      "variants": {
        "torch": {"api": "torch.nn.Sequential"},
        "keras": {"api": "keras.Sequential", "requires_plugin": "keras_sequential_pack"},
      },
    },
    "Linear": {
      "std_args": ["in", "out"],
      "variants": {
        "torch": {"api": "torch.nn.Linear"},
        # Naive mapping: preserves 2 args. In reality 'in' gets dropped or mapped to input_shape,
        # but for this test we only care that it became a Keras layer inside a list.
        "keras": {"api": "keras.layers.Dense", "args": {"out": "units"}},
      },
    },
    "ReLU": {"std_args": [], "variants": {"torch": {"api": "torch.nn.ReLU"}, "keras": {"api": "keras.layers.ReLU"}}},
  }

  def get_def(name):
    if "Sequential" in name:
      return ("Sequential", mappings["Sequential"])
    if "Linear" in name:
      return ("Linear", mappings["Linear"])
    if "ReLU" in name:
      return ("ReLU", mappings["ReLU"])
    return ("Generic", {"variants": {}})

  def resolve(aid, fw):
    if aid in mappings and fw == "keras":
      return mappings[aid]["variants"]["keras"]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}

  return mgr


def test_sequential_packing(keras_semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="keras")
  engine = ASTEngine(semantics=keras_semantics, config=config)

  result = engine.run(SOURCE_TORCH)

  assert result.success
  code = result.code

  # 1. Verify Packing: Sequential([...])
  assert "keras.Sequential([" in code.replace("\n", "").replace(" ", "")

  # 2. Verify Layers converted
  # We check for presence of Dense calls with arguments.
  # Note: The mock produces Dense(10, 20) because it maps 'out'->'units' but leaves 'in' positionally
  # if not explicitly ignored. This is acceptable for the Scope of checking Container packing.
  assert "keras.layers.Dense" in code
  assert "keras.layers.ReLU" in code
