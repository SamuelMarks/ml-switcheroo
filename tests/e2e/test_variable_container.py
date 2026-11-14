"""Test suite for the Variable Container module."""

import textwrap

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import register_hook
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_FLAX_VARIABLE: str = textwrap.dedent(
  "\n  import flax.nnx as nnx\n\n  class MyLayer(nnx.Module):\n    def __init__(self, rngs: nnx.Rngs):\n        self.param = nnx.Param(1.0)\n        self.var = nnx.Variable(2.0)\n        self.cache = nnx.Cache(3.0)\n"
)


@pytest.fixture(autouse=True)
def register_hooks() -> None:
  """Create a configured `None` fixture for variable container tests."""
  register_hook("nnx_param_to_torch")(transform_nnx_param)


@pytest.fixture
def semantics() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for variable container tests.

  Returns:
      Configured `semantics` fixture value (SemanticsManager).
  """
  mgr = SemanticsManager()
  mgr._key_origins["Variable"] = SemanticTier.NEURAL.value
  mgr._key_origins["Param"] = SemanticTier.NEURAL.value
  mgr._key_origins["Cache"] = SemanticTier.NEURAL.value

  def add(name: str, std_args: list[str], variants: dict[str, dict[str, str]]) -> None:
    """Execute `variants.values` to support add in variable container tests.

    Args:
        name: Identifier or operation name string.
        std_args: Input `std_args` parameter value (`list[str]`).
        variants: Per-framework operation variant mappings (`dict[str, dict[str, str]]`).
    """
    variants_json: dict[str, JsonValue] = {fw: dict[str, JsonValue](cfg) for fw, cfg in variants.items()}
    mgr.data[name] = {
      "std_args": list[JsonValue](std_args),
      "variants": variants_json,
    }
    for v in variants.values():
      mgr._reverse_index[v["api"]] = (
        name,
        mgr.data[name],
      )

  add(
    "Variable",
    ["value"],
    {
      "flax_nnx": {"api": "flax.nnx.Variable"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    },
  )
  add(
    "Param",
    ["value"],
    {
      "flax_nnx": {"api": "flax.nnx.Param"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    },
  )
  add(
    "Cache",
    ["value"],
    {
      "flax_nnx": {"api": "flax.nnx.Cache"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    },
  )
  mgr.framework_configs["flax_nnx"] = {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}}
  mgr.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}}
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
  if "torch" not in mgr._providers:
    mgr._providers["torch"] = {}
  mgr._providers["torch"][SemanticTier.NEURAL] = {"root": "torch", "sub": "nn", "alias": "nn"}
  return mgr


def test_flax_variable_to_torch(semantics: SemanticsManager) -> None:
  """Verify `RuntimeConfig` emits `nn.Parameter(1.0)` for flax variable to torch.

  Args:
      semantics: SemanticsManager instance or mock providing operation mappings.
  """
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_FLAX_VARIABLE)
  code: str = result.code
  assert result.success
  assert "nn.Parameter(1.0)" in code
  assert "nn.Parameter(2.0, requires_grad=False)" in code
  assert "nn.Parameter(3.0,requires_grad=False)" in code.replace(" ", "")
