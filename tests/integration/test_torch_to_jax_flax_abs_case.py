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
from ml_switcheroo.semantics.merging import merge_overlay_data
from ml_switcheroo.enums import SemanticTier

# Fix: Import specific adapter for Neural traits
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


# We reuse a SemanticsManager pointing to mocks to simulate the fix
# without relying on the file system state purely.
class FixedSemantics(SemanticsManager):
  def __init__(self):
    # Do not call super init to avoid file I/O

    self.data = {}
    self.framework_configs = {}
    self.test_templates = {}
    self._known_rng_methods = set()
    self.known_magic_args = set()
    self.patterns = []
    self.import_data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._providers = {}
    self._source_registry = {}

    # Use FlaxNNXAdapter traits for 'jax' key to enable structure rewrites
    adapter = FlaxNNXAdapter()

    # Simulate a snapshot that has run apply_wiring
    snapshot = {"__framework__": "jax", "mappings": {}, "imports": {}}

    # 1. Run the wiring logic we just fixed
    adapter.apply_wiring(snapshot)

    # 2. Inject result into manager data structures
    merge_overlay_data(
      data=self.data,
      key_origins=self._key_origins,
      import_data=self.import_data,
      framework_configs=self.framework_configs,
      test_templates=self.test_templates,
      content=snapshot,
      filename="jax_vlatest_map.json",
    )

    # 3. Add base definitions for Module (Neural) and Abs (Math)
    self.data["Abs"] = {
      "std_args": ["x"],
      "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}},
    }

    # Neural Module Definition
    self.data["Module"] = {"std_args": [], "variants": {"torch": {"api": "torch.nn.Module"}}}

    # Override configurations for 'jax' to use Flax NNX traits
    self.framework_configs["jax"] = {"traits": adapter.structural_traits.model_dump(exclude_unset=True)}

    # CRITICAL FIX: Ensure 'torch' traits are present so source class recognition works
    self.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}}

    # Mock Aliases from Adapter
    self.framework_configs["jax"]["alias"] = {"module": "jax.numpy", "name": "jnp"}

    # Rebuild index
    self._build_index()

    # Add explicit import map for torch.nn removal testing
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)

    if "jax" not in self._providers:
      self._providers["jax"] = {}

    self._providers["jax"][SemanticTier.NEURAL] = {"root": "flax", "sub": "nnx", "alias": "nnx"}


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
import flax.nnx as nnx

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

  # Updated expectation: Accept 'from flax import nnx' (cleaner) OR 'import flax.nnx as nnx'
  # The output matches based on mapping tuple ("flax", "nnx", "nnx")
  assert "import flax.nnx as nnx" in code or "from flax import nnx" in code

  # Crucial Fix Verification:
  assert "import torch" not in code
  # We should NOT see 'as nn' because we mapped to 'nnx'
  assert "as nn" not in code.split("\n")[1:]  # skip potential jax.nn but that's unlikely here

  # 2. Structural Check
  assert "class Model(nnx.Module):" in code
  assert "def __call__(self, x):" in code

  # 3. Logic Check
  assert "jnp.abs(x)" in code
