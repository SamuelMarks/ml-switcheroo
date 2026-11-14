"""Test suite for the Mlir Convnet Clean module."""

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager

INPUT_CODE: str = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


class MockConvNetSemantics(SemanticsManager):
  """Test double `MockConvNetSemantics` used in MLIR convnet clean tests."""

  def __init__(self) -> None:
    """Initializes the MockConvNetSemantics instance."""
    super().__init__(load_defaults=False)
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {
      "flax_nnx": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [["rngs", "flax.nnx.Rngs"]],
          "requires_super_init": True,
        },
        "alias": {"module": "flax.nnx", "name": "nnx"},
      },
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}},
    }
    self._add("Conv2d", "torch.nn.Conv2d", "flax.nnx.Conv", ["in", "out", "k"])
    self._add("Linear", "torch.nn.Linear", "flax.nnx.Linear", ["in", "out"])
    self._add("flatten", "torch.flatten", "flax.nnx.Flatten", ["x", "start_dim"])
    self._add("Module", "torch.nn.Module", "flax.nnx.Module", [])

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return set()

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Input `target_framework` parameter value (`str`).

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    return {}

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

  def _add(self, name: str, s_api: str, t_api: str, args: list[str]) -> None:
    """Execute `_add` on `MockConvNetSemantics` for MLIR convnet clean tests.

    Args:
        name: Identifier or operation name string.
        s_api: Input `s_api` parameter value (`str`).
        t_api: Input `t_api` parameter value (`str`).
        args: Positional argument specifications (`list[str]`).
    """
    variants: dict[str, JsonValue] = {"torch": {"api": s_api}, "flax_nnx": {"api": t_api}}
    std_args: list[JsonValue] = [*args]
    self.data[name] = {
      "std_args": std_args,
      "variants": variants,
    }
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = SemanticTier.NEURAL.value


def test_clean_mlir_generation() -> None:
  """Verify `RuntimeConfig` emits `super().__init__()` for clean MLIR generation."""
  semantics = MockConvNetSemantics()
  config = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    strict_mode=False,
    intermediate="mlir",
  )
  engine = ASTEngine(semantics=semantics, config=config)
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
