"""Test suite for the Mlir Roundtrip module."""

import typing
from unittest.mock import patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_CODE: str = "\nclass MyModel:\n    def forward(self, x):\n        return x\n"
CONVNET_SOURCE: str = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


class MockFlaxSemantics(SemanticsManager):
  """Test double `MockFlaxSemantics` used in MLIR roundtrip tests."""

  def __init__(self) -> None:
    """Initializes the MockFlaxSemantics instance."""
    super().__init__(load_defaults=False)
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs["torch"] = {
      "traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}
    }
    self.framework_configs["flax_nnx"] = {
      "traits": {
        "module_base": "flax.nnx.Module",
        "forward_method": "__call__",
        "inject_magic_args": [
          [
            "rngs",
            "flax.nnx.Rngs",
          ]
        ],
        "requires_super_init": False,
      },
      "alias": {"module": "flax.nnx", "name": "nnx"},
    }
    self._add_op("Conv2d", ["in", "out", "k"], "torch.nn.Conv2d", "flax.nnx.Conv")
    self._add_op("Linear", ["in", "out"], "torch.nn.Linear", "flax.nnx.Linear")
    self._add_op("Flatten", ["x", "dim"], "torch.flatten", "flax.nnx.Flatten")
    self._add_op("Module", [], "torch.nn.Module", "flax.nnx.Module")
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    self._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return set()

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Input `target_framework` parameter value (`str`).

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    if target_framework == "flax_nnx":
      return {"torch.nn": ("flax", "nnx", "nnx")}
    return {}

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get_definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    variants = self.data.get(abstract_id, {}).get("variants")
    if not isinstance(variants, dict):
      return None
    variant = variants.get(fw)
    return variant if isinstance(variant, dict) else None

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified.

    Args:
        _id: Input `_id` parameter value (`str`).

    Returns:
        Resulting `bool` from is verified.
    """
    return True

  def _add_op(self, name: str, args: list[str], s_api: str, t_api: str) -> None:
    """Execute `_add_op` on `MockFlaxSemantics` for MLIR roundtrip tests.

    Args:
        name: Identifier or operation name string.
        args: Positional argument specifications (`list[str]`).
        s_api: Input `s_api` parameter value (`str`).
        t_api: Input `t_api` parameter value (`str`).
    """
    variants: dict[str, JsonValue] = {"torch": {"api": s_api}, "flax_nnx": {"api": t_api}}
    std_args: list[JsonValue] = [*args]
    self.data[name] = {
      "std_args": std_args,
      "variants": variants,
    }
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = SemanticTier.NEURAL.value


@pytest.fixture
def engine_mlir() -> typing.Generator[ASTEngine, None, None]:
  """Create a configured `engine_mlir` fixture for MLIR roundtrip tests.

  Yields:
      Configured `engine_mlir` fixture value (typing.Generator[ASTEngine, None, None]).
  """
  config = RuntimeConfig(
    source_framework="mlir",
    target_framework="jax",
    strict_mode=False,
    intermediate="mlir",
  )
  with patch("ml_switcheroo.semantics.manager.SemanticsManager") as mock_mgr:
    mgr = mock_mgr.return_value
    mgr.get_framework_config.return_value = {}
    yield ASTEngine(semantics=mgr, config=config)


def test_mlir_bridge_activation(engine_mlir: ASTEngine) -> None:
  """Verify `engine_mlir.run` handles MLIR bridge activation as expected.

  Args:
      engine_mlir: Input `engine_mlir` parameter value (`ASTEngine`).
  """
  result: ConversionResult = engine_mlir.run('%0 = "sw.noop"()')
  assert result.success
  phase_names: list[str] = [str(e["description"]) for e in result.trace_events if e["type"] == TraceEventType.PHASE_START]
  assert "MLIR Ingest" in phase_names or "MLIR Bridge" in phase_names


def test_convnet_full_flow_mlir() -> None:
  """Verify `RuntimeConfig` emits `class ConvNet(nnx.Module):` for convnet full flow MLIR."""
  semantics = MockFlaxSemantics()
  config = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    strict_mode=False,
    intermediate="mlir",
  )
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(CONVNET_SOURCE)
  assert result.success, f"Errors: {result.errors}"
  code: str = result.code
  assert "class ConvNet(nnx.Module):" in code
  assert "def __init__(self, rngs: nnx.Rngs):" in code
  assert "nnx.Conv" in code
  assert "nnx.Linear" in code
  assert "rngs=rngs" in code
  assert "def __call__(" in code
  assert "nnx.Flatten" in code
  assert "flax.nnx" in code
