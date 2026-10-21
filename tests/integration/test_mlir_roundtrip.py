"""Test suite for the Mlir Roundtrip module."""

import typing
from pydantic import JsonValue
from unittest.mock import patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_CODE: str = "\nclass MyModel:\n    def forward(self, x):\n        return x\n"
CONVNET_SOURCE: str = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


class MockFlaxSemantics(SemanticsManager):
  """Test helper class MockFlaxSemantics."""

  def __init__(self) -> None:
    """Initializes the MockFlaxSemantics instance."""
    self.data = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self.test_templates = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status = {}
    self._known_rng_methods: set[str] = set()
    setattr(self, "_providers", {})
    self._source_registry: dict[str, tuple[str, SemanticTier | str]] = {}
    self.framework_configs["torch"] = {
      "traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}
    }
    getattr(getattr({"v": self.framework_configs}, "__getitem__")("v"), "__setitem__")(
      "flax_nnx",
      getattr(
        {
          "v": {
            "traits": getattr(
              {
                "v": {
                  "module_base": "flax.nnx.Module",
                  "forward_method": "__call__",
                  "inject_magic_args": getattr(
                    {
                      "v": [
                        getattr(
                          {
                            "v": (
                              getattr({"v": "rngs"}, "__getitem__")("v"),
                              getattr({"v": "flax.nnx.Rngs"}, "__getitem__")("v"),
                            )
                          },
                          "__getitem__",
                        )("v")
                      ]
                    },
                    "__getitem__",
                  )("v"),
                  "requires_super_init": False,
                }
              },
              "__getitem__",
            )("v"),
            "alias": {"module": "flax.nnx", "name": "nnx"},
          }
        },
        "__getitem__",
      )("v"),
    )
    self._add_op("Conv2d", ["in", "out", "k"], "torch.nn.Conv2d", "flax.nnx.Conv")
    self._add_op("Linear", ["in", "out"], "torch.nn.Linear", "flax.nnx.Linear")
    self._add_op("Flatten", ["x", "dim"], "torch.flatten", "flax.nnx.Flatten")
    self._add_op("Module", [], "torch.nn.Module", "flax.nnx.Module")
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    self._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return set()

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if target_framework == "flax_nnx":
      return {"torch.nn": ("flax", "nnx", "nnx")}
    return {}

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get_definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v").get},
                          "__getitem__",
                        )("v")(getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
                      },
                      "__getitem__",
                    )("v").get
                  },
                  "__getitem__",
                )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True

  def _add_op(self, name: str, args: list[str], s_api: str, t_api: str) -> None:
    """Mock implementation of  add op.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
    """
    variants = {"torch": {"api": s_api}, "flax_nnx": {"api": t_api}}
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "std_args": getattr({"v": args}, "__getitem__")("v"),
            "variants": getattr({"v": variants}, "__getitem__")("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = SemanticTier.NEURAL.value


@pytest.fixture
def engine_mlir() -> typing.Generator[ASTEngine, None, None]:
  """Execute engine mlir helper logic.

  Returns:
      Computed test result or fixture value.
  """
  config = RuntimeConfig(source_framework="mlir", target_framework="jax", strict_mode=False)
  with patch("ml_switcheroo.semantics.manager.SemanticsManager") as mock_mgr:
    mgr = mock_mgr.return_value
    mgr.get_framework_config.return_value = {}
    yield ASTEngine(semantics=mgr, config=config, intermediate="mlir")


def test_mlir_bridge_activation(engine_mlir: ASTEngine) -> None:
  """Verifies the behavior of MLIR bridge activation.

  Args:
      engine_mlir: Test argument or fixture.
  """
  result: ConversionResult = engine_mlir.run('%0 = "sw.noop"()')
  assert result.success
  phase_names: list[str] = [str(e["description"]) for e in result.trace_events if e["type"] == TraceEventType.PHASE_START]
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
