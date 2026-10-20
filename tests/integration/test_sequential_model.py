"""Test suite for the Sequential Model module."""

import typing
from pydantic import JsonValue

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSequentialSemantics(SemanticsManager):
  """Test helper class MockSequentialSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSequentialSemantics instance."""
    super().__init__()
    self.framework_configs["flax_nnx"] = {
      "alias": {"module": "flax.nnx", "name": "nnx"},
      "traits": {
        "module_base": "flax.nnx.Module",
        "forward_method": "__call__",
        "inject_magic_args": [("rngs", "flax.nnx.Rngs")],  # type: ignore[list-item]
      },
    }
    self._providers = {}
    self._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}
    self._source_registry = {}
    self._inject_op("Sequential", ["layers"], "torch.nn.Sequential", "flax.nnx.Sequential", SemanticTier.NEURAL)
    self._inject_op("Linear", ["in", "out"], "torch.nn.Linear", "flax.nnx.Linear", SemanticTier.NEURAL)
    self._inject_op("Flatten", ["start", "end"], "torch.nn.Flatten", "flax.nnx.Flatten", SemanticTier.NEURAL)
    self._inject_op("ReLU", [], "torch.nn.ReLU", "flax.nnx.relu", SemanticTier.NEURAL)

  def _inject_op(self, name: str, std_args: list[str], s_api: str, t_api: str, tier: SemanticTier) -> None:
    """Mock implementation of  inject op.

    Args:
        name: Test argument or fixture.
        std_args: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
        tier: Test argument or fixture.
    """
    if name not in self.data:
      self.data[name] = {"std_args": std_args, "variants": {}}  # type: ignore[dict-item]
    self.data[name]["variants"]["torch"] = {"api": s_api}  # type: ignore[index, call-overload]
    self.data[name]["variants"]["flax_nnx"] = {"api": t_api}  # type: ignore[index, call-overload]
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = tier.value
    self._source_registry[s_api] = ("torch", tier)

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map.

    Args:
        target_fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    result: dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]] = {}
    target_providers = self._providers.get(target_fw, {})
    for src_path, (src_fw, tier) in self._source_registry.items():
      if tier in target_providers:
        conf = target_providers[tier]
        result[src_path] = (conf["root"], conf["sub"], conf["alias"])  # type: ignore[assignment]
    return result


@pytest.fixture
def semantics_manager() -> MockSequentialSemantics:
  """Execute semantics manager helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return MockSequentialSemantics()


def test_sequential_container_transpilation(semantics_manager: MockSequentialSemantics) -> None:
  """Verifies the behavior of sequential container transpilation.

  Args:
      semantics_manager: Test argument or fixture.
  """
  source_code: str = "\nimport torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Flatten(),\n            nn.Linear(28 * 28, 512),\n            nn.ReLU(),\n            nn.Linear(512, 10)\n        )\n\n    def forward(self, x):\n        return self.net(x)\n"
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics_manager, config=config)
  result: ConversionResult = engine.run(source_code)
  code: str = result.code
  assert result.success
  assert "from flax import nnx" in code or "import flax.nnx as nnx" in code
  assert "class MLP(nnx.Module):" in code
  assert "nnx.Linear(512, 10, rngs=rngs)" in code
  assert "nnx.Flatten" in code
