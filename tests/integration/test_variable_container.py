"""Test suite for the Variable Container module."""

import pytest
import textwrap
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param

SOURCE_FLAX_VARIABLE = textwrap.dedent(
  "\n  import flax.nnx as nnx\n\n  class MyLayer(nnx.Module):\n    def __init__(self, rngs: nnx.Rngs):\n        self.param = nnx.Param(1.0)\n        self.var = nnx.Variable(2.0)\n        self.cache = nnx.Cache(3.0)\n"
)


@pytest.fixture(autouse=True)
def register_hooks():
  """Helper to register hooks."""
  _HOOKS["nnx_param_to_torch"] = transform_nnx_param


@pytest.fixture
def semantics():
  """Provides a mock semantics for testing."""
  mgr = SemanticsManager()
  mgr._key_origins["Variable"] = SemanticTier.NEURAL.value
  mgr._key_origins["Param"] = SemanticTier.NEURAL.value
  mgr._key_origins["Cache"] = SemanticTier.NEURAL.value

  def add(name, std_args, variants):
    """Adds ."""
    mgr.data[name] = {"std_args": std_args, "variants": variants}
    for fw, v in variants.items():
      mgr._reverse_index[v["api"]] = (name, mgr.data[name])

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


def test_flax_variable_to_torch(semantics):
  """Verifies the behavior of Flax variable to PyTorch."""
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_FLAX_VARIABLE)
  code = result.code
  assert result.success
  assert "nn.Parameter(1.0)" in code
  assert "nn.Parameter(2.0, requires_grad=False)" in code
  assert "nn.Parameter(3.0,requires_grad=False)" in code.replace(" ", "")
