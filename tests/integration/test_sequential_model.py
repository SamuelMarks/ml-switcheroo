"""
Integration Test for Sequential Container Mapping.

Verifies that `torch.nn.Sequential` is correctly transpiled to `flax.nnx.Sequential`,
and that its contents (layers) are recursively rewritten using proper aliases.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


class TestSemantics(SemanticsManager):
  """
  Overrides framework configs for testing to ensure traits are available
  without relying on external files or registry defaults that might be missing
  in isolated test environments.
  """

  def __init__(self):
    super().__init__()
    # Explicitly set traits for 'jax' (Flax NNX)
    self.framework_configs["flax_nnx"] = {
      "alias": {"module": "flax.nnx", "name": "nnx"},
      "traits": {
        "module_base": "flax.nnx.Module",
        "forward_method": "__call__",
        "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
      },
    }

    # Inject mappings necessary for this test
    # This ensures that even if JSON files are empty or incomplete in the test env,
    # the rewriter knows about these operations.

    # Sequential: torch.nn.Sequential -> flax.nnx.Sequential
    self._inject_op(
      "Sequential",
      ["layers"],
      "torch.nn.Sequential",
      "flax.nnx.Sequential",
      SemanticTier.NEURAL,
    )

    # Linear: torch.nn.Linear -> flax.nnx.Linear
    self._inject_op(
      "Linear",
      ["in_features", "out_features"],
      "torch.nn.Linear",
      "flax.nnx.Linear",
      SemanticTier.NEURAL,
    )

    # Flatten: torch.nn.Flatten -> flax.nnx.Flatten
    self._inject_op(
      "Flatten",
      ["start_dim", "end_dim"],
      "torch.nn.Flatten",
      "flax.nnx.Flatten",
      SemanticTier.NEURAL,
    )

    # ReLU: torch.nn.ReLU -> flax.nnx.relu
    # Mapped as NEURAL to verify uniform rngs injection for layer instantiation inside __init__
    self._inject_op("ReLU", [], "torch.nn.ReLU", "flax.nnx.relu", SemanticTier.NEURAL)

  def _inject_op(self, name, std_args, s_api, t_api, tier):
    if name not in self.data:
      self.data[name] = {"std_args": std_args, "variants": {}}

    if "variants" not in self.data[name]:
      self.data[name]["variants"] = {}

    self.data[name]["variants"]["torch"] = {"api": s_api}
    self.data[name]["variants"]["flax_nnx"] = {"api": t_api}

    self._reverse_index[s_api] = (name, self.data[name])
    # Key origins required for state injection logic
    self._key_origins[name] = tier.value

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
    target_framework="flax_nnx",
    strict_mode=False,  # Disabled to handle super() which fails strict resolution
  )

  engine = ASTEngine(semantics=semantics_manager, config=config)
  result = engine.run(source_code)

  # --- Verification ---

  assert result.success, f"Transpilation failed errors: {result.errors}"
  code = result.code

  # 1. Imports
  # Engine should inject 'nnx' based on usage
  assert "from flax import nnx" in code or "import flax.nnx as nnx" in code

  # 2. Class Definition (using alias)
  assert "class MLP(nnx.Module):" in code

  # 3. Sequential Container
  assert "self.net = nnx.Sequential(" in code

  # 4. Layers Structure
  # Flax NNX requires explicit state injection in __init__ calls if wrapped in class.
  # State injection is greedy for calls matching Neural Tier inside __init__.

  # Linear layers should have rngs injected and use alias
  assert "nnx.Linear(28 * 28, 512, rngs=rngs)" in code
  assert "nnx.Linear(512, 10, rngs=rngs)" in code

  # Flatten mappings
  assert "nnx.Flatten" in code

  # 5. Helper Check: ReLU mapping (functional lowercase in NNX often preferred, matches mock)
  # The Mock injects "flax.nnx.relu" which is aliased to "nnx.relu"
  assert "nnx.relu" in code

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
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics_manager, config=config)
  result = engine.run(source_code)

  code = result.code

  # Linear gets rngs
  assert "nnx.Linear(10, 10, rngs=rngs)" in code

  # Sequential gets rngs if it's Neural Tier (it is)
  # The output from previous runs shows it appends rngs=rngs to Sequential too
  assert "self.seq = nnx.Sequential(*layers, rngs=rngs)" in code
