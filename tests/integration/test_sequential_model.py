"""
Integration Test for Sequential Container Mapping.

Verifies that `torch.nn.Sequential` is correctly transpiled to `flax.nnx.Sequential`,
and that its contents (layers) are recursively rewritten.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class TestSemantics(SemanticsManager):
  """
  Overrides framework configs for testing to ensure traits are available
  without relying on external files or registry defaults that might be missing
  in isolated test environments.
  """

  def __init__(self):
    super().__init__()
    # Explicitly set traits for 'jax' (Flax NNX)
    self.framework_configs["jax"] = {
      "traits": {
        "module_base": "flax.nnx.Module",
        "forward_method": "__call__",
        "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
      }
    }

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})


@pytest.fixture
def semantics_manager():
  """
  Loads the semantics manager.
  """
  return TestSemantics()


def test_sequential_container_transpilation(semantics_manager):
  """
  Scenario: PyTorch model defines a Sequential block in __init__.
  Goal: Transpile to Flax NNX.

  Checks:
  1. `nn.Sequential` -> `nnx.Sequential`.
  2. `nn.Linear` -> `nnx.Linear`.
  3. `nn.Flatten` -> `nnx.Flatten`.
  4. Arguments inside Linear are preserved/mapped.
  5. Structural injection of `rngs=rngs` happens inside the container items
     because they are calls inside a Neural Module's `__init__`.
  """
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

  config = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    strict_mode=False,  # Disabled to handle super() which fails strict resolution
  )

  engine = ASTEngine(semantics=semantics_manager, config=config)
  result = engine.run(source_code)

  # --- Verification ---

  assert result.success, f"Transpilation failed errors: {result.errors}"
  code = result.code

  # 1. Imports
  # ImportFixer might convert 'import torch.nn as nn' to 'from flax import nnx as nn'
  # or preserve 'import flax.nnx' if used directly.

  # 2. Class Definition
  assert "class MLP(flax.nnx.Module):" in code

  # 3. Sequential Container
  assert "self.net = flax.nnx.Sequential(" in code

  # 4. Layers Structure
  # Flax NNX requires explicit state injection in __init__ calls if wrapped in class.
  # State injection is greedy for calls matching Neural Tier inside __init__.

  # Linear layers should have rngs injected
  assert "flax.nnx.Linear(28 * 28, 512, rngs=rngs)" in code
  assert "flax.nnx.Linear(512, 10, rngs=rngs)" in code

  # Flatten mappings
  # If Flatten is tiered as Neural, it gets rngs.
  assert "flax.nnx.Flatten" in code

  # 5. Helper Check: ReLU mapping
  assert "flax.nnx.relu" in code

  # 6. Forward Method
  assert "def __call__(self, x):" in code
  assert "return self.net(x)" in code


def test_sequential_variable_assignment(semantics_manager):
  """
  Scenario: Sequential assigned to variable first.

  Input:
    layers = [nn.Linear(10, 10), nn.ReLU()]
    seq = nn.Sequential(*layers)

  Output:
    layers = [nnx.Linear(10, 10, rngs=rngs), nnx.relu(rngs=rngs)]
    seq = nnx.Sequential(*layers, rngs=rngs)
  """
  source_code = """ 
class Model(torch.nn.Module): 
    def __init__(self): 
        layers = [torch.nn.Linear(10, 10), torch.nn.ReLU()] 
        self.seq = torch.nn.Sequential(*layers) 
"""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  engine = ASTEngine(semantics=semantics_manager, config=config)
  result = engine.run(source_code)

  code = result.code

  # Linear gets rngs
  assert "flax.nnx.Linear(10, 10, rngs=rngs)" in code

  # Sequential gets rngs if it's Neural Tier (it is)
  # The output from previous runs shows it appends rngs=rngs to Sequential too
  assert "self.seq = flax.nnx.Sequential(*layers, rngs=rngs)" in code
