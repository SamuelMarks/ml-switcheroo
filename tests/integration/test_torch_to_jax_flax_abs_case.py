"""
Integration test for the specific user reported case (torch.abs -> jnp.abs).

Verifies:
1. `import torch` is removed.
2. `import torch.nn` is removed (because `nn` usage is swapped to `nnx` imports).
3. `torch.abs(x)` becomes `jnp.abs(x)`.
4. `nn.Module` becomes `nnx.Module`.
"""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks.jax import JaxAdapter


# We use a SemanticsManager pointing to mocks to simulate the fix
# without relying on the file system state purely.
class FixedSemantics(SemanticsManager):
  def __init__(self):
    super().__init__()
    # Apply the logic from the corrected Adapter directly to memory
    # to prove the fix works given the snapshot structure we expect
    adapter = JaxAdapter()

    # Simulate a snapshot that has run apply_wiring
    snapshot = {"__framework__": "jax", "mappings": {}, "imports": {}}

    # 1. Run the wiring logic we just fixed
    adapter.apply_wiring(snapshot)

    # 2. Inject result into manager data structures
    self._merge_overlay(snapshot, "jax_vlatest_map.json")

    # 3. Add base definitions for Module (Neural) and Abs (Math)
    self.data["Abs"] = {
      "std_args": ["x"],
      "variants": {"torch": {"api": "torch.abs"}, "jax": snapshot.get("mappings", {}).get("Abs")},
    }

    # Neural Module Definition
    self.data["Module"] = {"std_args": [], "variants": {"torch": {"api": "torch.nn.Module"}}}

    # Override configurations
    self.framework_configs["jax"] = {"traits": adapter.structural_traits.model_dump(exclude_unset=True)}

    # Mock Aliases from Adapter
    self.framework_configs["jax"]["alias"] = {"module": "jax.numpy", "name": "jnp"}

    # Rebuild index
    self._build_index()

    # Add explicit import map for torch.nn removal testing
    # NOTE: We do NOT add torch.nn -> flax.nnx here, confirming the fix


def test_specific_abs_conversion():
  input_torch = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.abs(x)
"""
  output_jax_flax = """
import jax.numpy as jnp
from flax import nnx

class Model(nnx.Module):
    def __call__(self, x):
        return jnp.abs(x)
"""

  semantics = FixedSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(input_torch)

  assert result.success
  code = result.code

  # 1. Imports Check
  assert "import jax.numpy as jnp" in code
  assert "from flax import nnx" in code

  # Crucial Fix Verification:
  assert "import torch" not in code
  assert "import flax.nnx as nn" not in code

  # 2. Structural Check
  assert "class Model(nnx.Module):" in code
  assert "def __call__(self, x):" in code

  # 3. Logic Check
  assert "jnp.abs(x)" in code

  assert code == output_jax_flax
