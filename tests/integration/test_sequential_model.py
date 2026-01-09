"""
Integration Test for Sequential Container Mapping.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


class MockSequentialSemantics(SemanticsManager):
  def __init__(self):
    super().__init__()

    # Configure Traits & Alias for Flax
    self.framework_configs["flax_nnx"] = {
      "alias": {"module": "flax.nnx", "name": "nnx"},
      "traits": {
        "module_base": "flax.nnx.Module",
        "forward_method": "__call__",
        "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
      },
    }

    # Import Provider Config to allow ImportFixer to generate "from flax import nnx"
    self._providers = {}
    self._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}
    self._source_registry = {}

    # Mappings
    self._inject_op("Sequential", ["layers"], "torch.nn.Sequential", "flax.nnx.Sequential", SemanticTier.NEURAL)
    self._inject_op("Linear", ["in", "out"], "torch.nn.Linear", "flax.nnx.Linear", SemanticTier.NEURAL)
    self._inject_op("Flatten", ["start", "end"], "torch.nn.Flatten", "flax.nnx.Flatten", SemanticTier.NEURAL)
    self._inject_op("ReLU", [], "torch.nn.ReLU", "flax.nnx.relu", SemanticTier.NEURAL)

  def _inject_op(self, name, std_args, s_api, t_api, tier):
    if name not in self.data:
      self.data[name] = {"std_args": std_args, "variants": {}}
    self.data[name]["variants"]["torch"] = {"api": s_api}
    self.data[name]["variants"]["flax_nnx"] = {"api": t_api}
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = tier.value
    # Register source for ImportFixer pruning/detection
    self._source_registry[s_api] = ("torch", tier)

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  # Must override import map to use provider logic
  def get_import_map(self, target_fw: str):
    result = {}
    target_providers = self._providers.get(target_fw, {})
    for src_path, (src_fw, tier) in self._source_registry.items():
      if tier in target_providers:
        conf = target_providers[tier]
        result[src_path] = (conf["root"], conf["sub"], conf["alias"])
    return result


@pytest.fixture
def semantics_manager():
  return MockSequentialSemantics()


def test_sequential_container_transpilation(semantics_manager):
  source_code = """ 
import torch
import torch.nn as nn

class MLP(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.net = nn.Sequential( 
            nn.Flatten(), 
            nn.Linear(28 * 28, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10) 
        ) 

    def forward(self, x): 
        return self.net(x) 
"""
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics_manager, config=config)
  result = engine.run(source_code)
  code = result.code

  assert result.success

  # New InjectionMixin logic prefers 'from flax import nnx as nnx' or similar if subcomponent defined
  assert "from flax import nnx" in code or "import flax.nnx as nnx" in code

  assert "class MLP(nnx.Module):" in code
  assert "nnx.Linear(512, 10, rngs=rngs)" in code
  assert "nnx.Flatten" in code
