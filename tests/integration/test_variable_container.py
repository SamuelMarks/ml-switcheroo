"""Test suite for the Variable Container module."""

import textwrap
from pydantic import JsonValue

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_FLAX_VARIABLE: str = textwrap.dedent(
  "\n  import flax.nnx as nnx\n\n  class MyLayer(nnx.Module):\n    def __init__(self, rngs: nnx.Rngs):\n        self.param = nnx.Param(1.0)\n        self.var = nnx.Variable(2.0)\n        self.cache = nnx.Cache(3.0)\n"
)


@pytest.fixture(autouse=True)
def register_hooks() -> None:
  """Helper to register hooks."""
  getattr(getattr({"v": _HOOKS}, "__getitem__")("v"), "__setitem__")(
    "nnx_param_to_torch", getattr({"v": transform_nnx_param}, "__getitem__")("v")
  )


@pytest.fixture
def semantics() -> SemanticsManager:
  """Execute semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr = SemanticsManager()
  mgr._key_origins["Variable"] = SemanticTier.NEURAL.value
  mgr._key_origins["Param"] = SemanticTier.NEURAL.value
  mgr._key_origins["Cache"] = SemanticTier.NEURAL.value

  def add(name: str, std_args: list[str], variants: dict[str, JsonValue]) -> None:
    """Adds .

    Args:
        name: Test argument or fixture.
        std_args: Test argument or fixture.
        variants: Test argument or fixture.
    """
    getattr(getattr({"v": getattr({"v": mgr}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "std_args": getattr({"v": std_args}, "__getitem__")("v"),
            "variants": getattr({"v": variants}, "__getitem__")("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    for fw, v in variants.items():
      getattr(getattr({"v": getattr({"v": mgr}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__")(
        getattr({"v": v}, "__getitem__")("v")["api"],
        getattr(
          {
            "v": (
              getattr({"v": name}, "__getitem__")("v"),
              getattr(
                {"v": getattr({"v": getattr({"v": mgr}, "__getitem__")("v").data}, "__getitem__")("v")[name]},
                "__getitem__",
              )("v"),
            )
          },
          "__getitem__",
        )("v"),
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
  """Verifies the behavior of Flax variable to PyTorch.

  Args:
      semantics: Test argument or fixture.
  """
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_FLAX_VARIABLE)
  code: str = result.code
  assert result.success
  assert "nn.Parameter(1.0)" in code
  assert "nn.Parameter(2.0, requires_grad=False)" in code
  assert "nn.Parameter(3.0,requires_grad=False)" in code.replace(" ", "")
