"""Test suite for the Mlir Convnet Clean module."""

import typing
from pydantic import JsonValue

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

INPUT_CODE: str = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


class MockConvNetSemantics(SemanticsManager):
  """Test helper class MockConvNetSemantics."""

  def __init__(self) -> None:
    """Initializes the MockConvNetSemantics instance."""
    self.data = {}
    self._providers = {}
    self._source_registry = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status = {}
    self._known_rng_methods: set[str] = set()
    self.test_templates = {}
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "framework_configs",
      getattr(
        {
          "v": {
            "flax_nnx": getattr(
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
                        "requires_super_init": True,
                      }
                    },
                    "__getitem__",
                  )("v"),
                  "alias": {"module": "flax.nnx", "name": "nnx"},
                }
              },
              "__getitem__",
            )("v"),
            "torch": {
              "traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}
            },
          }
        },
        "__getitem__",
      )("v"),
    )
    self._add("Conv2d", "torch.nn.Conv2d", "flax.nnx.Conv", ["in", "out", "k"])
    self._add("Linear", "torch.nn.Linear", "flax.nnx.Linear", ["in", "out"])
    self._add("flatten", "torch.flatten", "flax.nnx.Flatten", ["x", "start_dim"])
    self._add("Module", "torch.nn.Module", "flax.nnx.Module", [])

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return set()

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}

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

  def _add(self, name: str, s_api: str, t_api: str, args: list[str]) -> None:
    """Mock implementation of  add.

    Args:
        name: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
        args: Test argument or fixture.
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


def test_clean_mlir_generation() -> None:
  """Verifies the behavior of clean MLIR generation."""
  semantics = MockConvNetSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config, intermediate="mlir")
  result: ConversionResult = engine.run(INPUT_CODE)
  assert result.success
  code: str = result.code
  print("\n[Generated Code via MLIR]:")
  print(code)
  import re

  assert not re.search("_[a-z0-9_]+\\s+=\\s+super\\(\\)\\.__init__\\(\\)", code)
  assert "super().__init__()" in code
  for line in code.splitlines():
    if "self.fc(" in line:
      assert "Flatten(" not in line, f"Found nested Flatten in: {line}"
      assert "self.conv(" not in line, f"Found nested Conv in: {line}"
  assert " = self.conv" in code
  assert " = flax.nnx.Flatten" in code or " = nnx.Flatten" in code
