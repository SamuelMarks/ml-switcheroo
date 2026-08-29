"""Test suite for the Keras Sequential module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_TORCH: str = "\nimport torch.nn as nn\ndef get_model():\n    model = nn.Sequential(\n        nn.Linear(10, 20),\n        nn.ReLU(),\n        nn.Linear(20, 1)\n    )\n    return model\n"


@pytest.fixture
def keras_semantics() -> MagicMock:
  """Docstring."""
  _HOOKS["keras_sequential_pack"] = transform_keras_sequential
  mgr = MagicMock(spec=SemanticsManager)
  mappings: dict[str, typing.Any] = {
    "Sequential": {
      "std_args": ["layers"],
      "variants": {
        "torch": {"api": "torch.nn.Sequential"},
        "keras": {"api": "keras.Sequential", "requires_plugin": "keras_sequential_pack"},
      },
    },
    "Linear": {
      "std_args": ["in", "out"],
      "variants": {"torch": {"api": "torch.nn.Linear"}, "keras": {"api": "keras.layers.Dense", "args": {"out": "units"}}},
    },
    "ReLU": {"std_args": [], "variants": {"torch": {"api": "torch.nn.ReLU"}, "keras": {"api": "keras.layers.ReLU"}}},
  }

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    if "Sequential" in name:
      return ("Sequential", mappings["Sequential"])
    if "Linear" in name:
      return ("Linear", mappings["Linear"])
    if "ReLU" in name:
      return ("ReLU", mappings["ReLU"])
    return ("Generic", {"variants": {}})

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    if aid in mappings and fw == "keras":
      return mappings[aid]["variants"]["keras"]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}
  return mgr


def test_sequential_packing(keras_semantics: MagicMock) -> None:
  """Verifies the behavior of sequential packing."""
  config = RuntimeConfig(source_framework="torch", target_framework="keras")
  engine = ASTEngine(semantics=keras_semantics, config=config)  # type: ignore
  result: ConversionResult = engine.run(SOURCE_TORCH)
  assert result.success
  code: str = result.code
  assert "keras.Sequential([" in code.replace("\n", "").replace(" ", "")
  assert "keras.layers.Dense" in code
  assert "keras.layers.ReLU" in code
