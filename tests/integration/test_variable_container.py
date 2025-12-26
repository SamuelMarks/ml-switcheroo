"""
Integration Test for Variable/Parameter State Conversion.
"""

import pytest
import textwrap
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param

SOURCE_FLAX_VARIABLE = textwrap.dedent("""
  import flax.nnx as nnx
  
  class MyLayer(nnx.Module): 
    def __init__(self, rngs: nnx.Rngs): 
        self.param = nnx.Param(1.0) 
        self.var = nnx.Variable(2.0) 
        self.cache = nnx.Cache(3.0) 
""")


@pytest.fixture(autouse=True)
def register_hooks():
  _HOOKS["nnx_param_to_torch"] = transform_nnx_param


@pytest.fixture
def semantics():
  mgr = SemanticsManager()

  # Inject definitions manually to ensure test isolation from files
  mgr._key_origins["Variable"] = SemanticTier.NEURAL.value
  mgr._key_origins["Param"] = SemanticTier.NEURAL.value
  mgr._key_origins["Cache"] = SemanticTier.NEURAL.value

  # Helper
  def add(name, std_args, variants):
    mgr.data[name] = {"std_args": std_args, "variants": variants}
    for fw, v in variants.items():
      mgr._reverse_index[v["api"]] = (name, mgr.data[name])

  # Variable
  add(
    "Variable",
    ["value"],
    {
      "flax_nnx": {"api": "flax.nnx.Variable"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    },
  )

  # Param
  add(
    "Param",
    ["value"],
    {
      "flax_nnx": {"api": "flax.nnx.Param"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    },
  )

  # Cache
  add(
    "Cache",
    ["value"],
    {
      "flax_nnx": {"api": "flax.nnx.Cache"},
      "torch": {"api": "torch.nn.Parameter", "requires_plugin": "nnx_param_to_torch"},
    },
  )

  # Module base injection for detection
  mgr.framework_configs["flax_nnx"] = {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}}

  # Torch config for import collapsing (e.g. torch.nn -> nn)
  mgr.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}}

  # Mock import data for torch.nn to trigger aliasing if used by ImportFixer
  # New Logic
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)

  if "torch" not in mgr._providers:
    mgr._providers["torch"] = {}

  # Define provider for Torch self-conversion (preservation/aliasing)
  mgr._providers["torch"][SemanticTier.NEURAL] = {"root": "torch", "sub": "nn", "alias": "nn"}

  return mgr


def test_flax_variable_to_torch(semantics):
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_FLAX_VARIABLE)
  code = result.code

  assert result.success

  # Updated assertions to allow 'nn.Parameter' alias which provides cleaner code
  # We check for substring presence of the constructor call

  # 1. nnx.Param -> nn.Parameter(1.0)
  assert "nn.Parameter(1.0)" in code

  # 2. nnx.Variable -> nn.Parameter(2.0, requires_grad=False)
  assert "nn.Parameter(2.0, requires_grad=False)" in code

  # 3. nnx.Cache -> nn.Parameter(3.0, requires_grad=False)
  # Using replace to ignore whitespace variations
  assert "nn.Parameter(3.0,requires_grad=False)" in code.replace(" ", "")
