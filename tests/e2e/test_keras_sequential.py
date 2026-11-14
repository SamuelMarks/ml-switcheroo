"""Test suite for the Keras Sequential module."""

from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import register_hook
from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_TORCH: str = "\nimport torch.nn as nn\ndef get_model():\n    model = nn.Sequential(\n        nn.Linear(10, 20),\n        nn.ReLU(),\n        nn.Linear(20, 1)\n    )\n    return model\n"


@pytest.fixture
def keras_semantics() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for keras sequential tests.

  Returns:
      Configured `keras_semantics` fixture value (SemanticsManager).
  """
  register_hook("keras_sequential_pack")(transform_keras_sequential)
  mgr = MagicMock(spec=SemanticsManager)
  mappings: dict[str, dict[str, JsonValue]] = {
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
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    if "Sequential" in name:
      return ("Sequential", mappings["Sequential"])
    if "Linear" in name:
      return ("Linear", mappings["Linear"])
    if "ReLU" in name:
      return ("ReLU", mappings["ReLU"])
    return ("Generic", {"variants": {}})

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `resolve` helper in keras sequential tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    if aid in mappings and fw == "keras":
      variants = mappings[aid].get("variants")
      if isinstance(variants, dict):
        keras_variant = variants.get("keras")
        if isinstance(keras_variant, dict):
          return keras_variant
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}
  return mgr


def test_sequential_packing(keras_semantics: MagicMock) -> None:
  """Verify `RuntimeConfig` emits `keras.Sequential([` for sequential packing.

  Args:
      keras_semantics: Input `keras_semantics` parameter value (`MagicMock`).
  """
  config = RuntimeConfig(source_framework="torch", target_framework="keras")
  engine = ASTEngine(semantics=keras_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH)
  assert result.success
  code: str = result.code
  assert "keras.Sequential([" in code.replace("\n", "").replace(" ", "")
  assert "keras.layers.Dense" in code
  assert "keras.layers.ReLU" in code
