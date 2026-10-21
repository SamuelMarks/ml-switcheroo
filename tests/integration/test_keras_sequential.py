"""Test suite for the Keras Sequential module."""

from pydantic import JsonValue
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_TORCH: str = "\nimport torch.nn as nn\ndef get_model():\n    model = nn.Sequential(\n        nn.Linear(10, 20),\n        nn.ReLU(),\n        nn.Linear(20, 1)\n    )\n    return model\n"


@pytest.fixture
def keras_semantics() -> SemanticsManager:
  """Execute keras semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  getattr(getattr({"v": _HOOKS}, "__getitem__")("v"), "__setitem__")(
    "keras_sequential_pack", getattr({"v": transform_keras_sequential}, "__getitem__")("v")
  )
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
      "variants": {"torch": {"api": "torch.nn.Linear"}, "keras": {"api": "keras.layers.Dense", "args": {"out": "units"}}},
    },
    "ReLU": {"std_args": [], "variants": {"torch": {"api": "torch.nn.ReLU"}, "keras": {"api": "keras.layers.ReLU"}}},
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "Sequential" in name:
      _ret_v_1: tuple[str, dict[str, JsonValue]] | None = getattr(
        {
          "v": (
            getattr({"v": "Sequential"}, "__getitem__")("v"),
            getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["Sequential"]}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v")
      return _ret_v_1
    if "Linear" in name:
      _ret_v_2: tuple[str, dict[str, JsonValue]] | None = getattr(
        {
          "v": (
            getattr({"v": "Linear"}, "__getitem__")("v"),
            getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["Linear"]}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v")
      return _ret_v_2
    if "ReLU" in name:
      _ret_v_3: tuple[str, dict[str, JsonValue]] | None = getattr(
        {
          "v": (
            getattr({"v": "ReLU"}, "__getitem__")("v"),
            getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["ReLU"]}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v")
      return _ret_v_3
    return ("Generic", {"variants": {}})

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid in mappings and fw == "keras":
      _ret_v_4: dict[str, JsonValue] | None = getattr(
        {
          "v": getattr(
            {"v": getattr({"v": getattr({"v": mappings}, "__getitem__")("v")[aid]}, "__getitem__")("v")["variants"]},
            "__getitem__",
          )("v")["keras"]
        },
        "__getitem__",
      )("v")
      return _ret_v_4
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}
  return mgr


def test_sequential_packing(keras_semantics: MagicMock) -> None:
  """Verifies the behavior of sequential packing.

  Args:
      keras_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="keras")
  engine = ASTEngine(semantics=keras_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH)
  assert result.success
  code: str = result.code
  assert "keras.Sequential([" in code.replace("\n", "").replace(" ", "")
  assert "keras.layers.Dense" in code
  assert "keras.layers.ReLU" in code
