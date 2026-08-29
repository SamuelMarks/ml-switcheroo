"""Test suite for the Mlir Roundtrip module."""

import typing
from unittest.mock import patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_CODE: str = "\nclass MyModel:\n    def forward(self, x):\n        return x\n"
CONVNET_SOURCE: str = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


class MockFlaxSemantics(SemanticsManager):  # type: ignore[misc]
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockFlaxSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self.test_templates: dict[str, typing.Any] = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, tuple[str, SemanticTier]] = {}
    self.framework_configs["torch"] = {
      "traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}
    }
    self.framework_configs["flax_nnx"] = {
      "traits": {
        "module_base": "flax.nnx.Module",
        "forward_method": "__call__",
        "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
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
    """Mock implementation of get all rng methods."""
    return set()

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map."""
    if target_fw == "flax_nnx":
      return {"torch.nn": ("flax", "nnx", "nnx")}
    return {}

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return typing.cast(dict[str, typing.Any], self.framework_configs.get(framework, {}))

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get_definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified."""
    return True

  def _add_op(self, name: str, args: list[str], s_api: str, t_api: str) -> None:
    """Mock implementation of  add op."""
    variants: dict[str, typing.Any] = {"torch": {"api": s_api}, "flax_nnx": {"api": t_api}}
    self.data[name] = {"std_args": args, "variants": variants}
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = SemanticTier.NEURAL.value


@pytest.fixture
def engine_mlir() -> typing.Generator[ASTEngine, None, None]:
  """Docstring."""
  config = RuntimeConfig(source_framework="mlir", target_framework="jax", strict_mode=False)
  with patch("ml_switcheroo.semantics.manager.SemanticsManager") as mock_mgr:
    mgr = mock_mgr.return_value
    mgr.get_framework_config.return_value = {}
    yield ASTEngine(semantics=mgr, config=config, intermediate="mlir")


def test_mlir_bridge_activation(engine_mlir: ASTEngine) -> None:
  """Verifies the behavior of MLIR bridge activation."""
  result: ConversionResult = engine_mlir.run('%0 = "sw.noop"()')
  assert result.success
  phase_names: list[str] = [
    typing.cast(str, e["description"]) for e in result.trace_events if e["type"] == TraceEventType.PHASE_START
  ]
  assert "MLIR Ingest" in phase_names or "MLIR Bridge" in phase_names


def test_convnet_full_flow_mlir() -> None:
  """Verifies the behavior of convnet full flow MLIR."""
  semantics = MockFlaxSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config, intermediate="mlir")
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
