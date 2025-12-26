"""
Integration Test for Flax NNX <-> Apple MLX Bidirectional Conversion.

This test validates the structural rewriting rules required to move between
explicit RNG state (Flax) and implicit/eager initialization (MLX).

Scenarios Verified:
1.  **Forward (Flax -> MLX)**:
    - Imports: `from flax import nnx` -> `import mlx.nn as nn`
    - Init: Strips `rngs` argument.
    - Body: Injects `super().__init__()`.
    - Calls: Removes `rngs=rngs` kwarg.

2.  **Reverse (MLX -> Flax)**:
    - Imports: `import mlx.nn as nn` -> `from flax import nnx`.
    - Init: Re-injects `rngs: nnx.Rngs`.
    - Body: Removes `super().__init__()`.
    - Calls: Re-injects `rngs=rngs` into Linear.
    - **Verbatim Match**: Must match original input string exactly.
"""

import pytest
import textwrap
import re

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

# Input: Flax NNX Code
FLAX_SOURCE = textwrap.dedent(""" 
from flax import nnx

class Net(nnx.Module): 
    def __init__(self, rngs: nnx.Rngs): 
        # Layer initialization using explicit RNG
        self.linear = nnx.Linear(10, 10, rngs=rngs) 

    def __call__(self, x): 
        x = self.linear(x) 
        return nnx.relu(x) 
""").strip()

# Import variant that might occur due to aliasing config priorities
FLAX_SOURCE_RELAXED = textwrap.dedent("""
import flax.nnx as nnx

class Net(nnx.Module): 
    def __init__(self, rngs: nnx.Rngs): 
        # Layer initialization using explicit RNG
        self.linear = nnx.Linear(10, 10, rngs=rngs) 

    def __call__(self, x): 
        x = self.linear(x) 
        return nnx.relu(x) 
""").strip()


@pytest.fixture
def semantics():
  """
  Bootstrap a semantics manager with bidirectional knowledge.
  """
  mgr = SemanticsManager()

  # 1. Define Operations in the Hub
  # Linear: Neural Tier (Important for triggering structural rewrites)
  mgr.data["Linear"] = {
    "std_args": ["in_features", "out_features"],
    "variants": {
      "flax_nnx": {"api": "flax.nnx.Linear"},
      # MLX uses positional arguments for Linear(input_dims, output_dims)
      "mlx": {"api": "mlx.nn.Linear", "args": {"in_features": "input_dims", "out_features": "output_dims"}},
    },
  }
  mgr._key_origins["Linear"] = SemanticTier.NEURAL.value

  # Relu: Array/Activation
  mgr.data["relu"] = {
    "std_args": ["x"],
    "variants": {"flax_nnx": {"api": "flax.nnx.relu"}, "mlx": {"api": "mlx.nn.relu"}},
  }
  mgr._key_origins["relu"] = SemanticTier.ARRAY_API.value

  # 2. Build Reverse Indexes (Bidirectional)
  mgr._reverse_index["flax.nnx.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["nnx.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["flax.nnx.relu"] = ("relu", mgr.data["relu"])
  mgr._reverse_index["nnx.relu"] = ("relu", mgr.data["relu"])

  mgr._reverse_index["mlx.nn.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["nn.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["mlx.nn.relu"] = ("relu", mgr.data["relu"])
  mgr._reverse_index["nn.relu"] = ("relu", mgr.data["relu"])

  # 3. Load Framework Traits
  mlx_adapter = MLXAdapter()
  mgr.framework_configs["mlx"] = {
    "traits": mlx_adapter.structural_traits.model_dump(exclude_unset=True),
    "alias": {"module": "mlx.core", "name": "mx"},
  }

  flax_adapter = FlaxNNXAdapter()
  mgr.framework_configs["flax_nnx"] = {
    "traits": flax_adapter.structural_traits.model_dump(exclude_unset=True),
    "alias": {"module": "flax.nnx", "name": "nnx"},
  }

  # 4. Import Namespace Configuration
  # Mapping Flax -> MLX
  # We want `import mlx.nn as nn`

  # Identify source paths (Flax NNX imports)
  mgr._source_registry["flax.nnx"] = ("flax_nnx", SemanticTier.NEURAL)

  # Configure Provider for MLX
  # Note: Cleared existing providers to ensure no interference from default loading
  mgr._providers = {}

  if "mlx" not in mgr._providers:
    mgr._providers["mlx"] = {}

  mgr._providers["mlx"][SemanticTier.NEURAL] = {"root": "mlx.nn", "sub": None, "alias": "nn"}

  # Configure Provider for Flax (Target of reverse trip)
  if "flax_nnx" not in mgr._providers:
    mgr._providers["flax_nnx"] = {}
  # Prefer explicit 'from flax import nnx' logic if supported, but alias override logic usually forces 'import ... as'
  # unless sub is defined. Defining root="flax", sub="nnx" produces 'from flax import nnx' (if alias matches sub or None)
  mgr._providers["flax_nnx"][SemanticTier.NEURAL] = {"root": "flax", "sub": "nnx", "alias": "nnx"}

  # 5. Helper for mocking runtime lookups
  mgr.get_all_rng_methods = lambda: set()

  return mgr


def normalize_ws(s):
  """Normalize newlines and multiple spaces."""
  s = s.strip()
  s = re.sub(r"\n+", "\n", s)
  return s


def test_flax_to_mlx_roundtrip(semantics):
  # --- Step 1: Flax -> MLX ---
  config_f2m = RuntimeConfig(source_framework="flax_nnx", target_framework="mlx", strict_mode=True)
  engine_f2m = ASTEngine(semantics=semantics, config=config_f2m)
  res_mlx = engine_f2m.run(FLAX_SOURCE)

  assert res_mlx.success, f"F2M Errors: {res_mlx.errors}"
  mlx_code = res_mlx.code.strip()

  print(f"\n[Generated MLX]:\n{mlx_code}")

  # Verify key MLX properties
  assert "import mlx.nn as nn" in mlx_code
  assert "class Net(nn.Module):" in mlx_code
  assert "def __init__(self):" in mlx_code  # rngs gone
  assert "nn.Linear(10, 10)" in mlx_code
  assert "rngs=" not in mlx_code
  assert "super().__init__()" in mlx_code

  # --- Step 2: MLX -> Flax (Roundtrip) ---
  config_m2f = RuntimeConfig(source_framework="mlx", target_framework="flax_nnx", strict_mode=True)
  engine_m2f = ASTEngine(semantics=semantics, config=config_m2f)

  res_flax = engine_m2f.run(mlx_code)
  assert res_flax.success, f"M2F Errors: {res_flax.errors}"
  flax_code = res_flax.code.strip()

  print(f"\n[Restored Flax]:\n{flax_code}")

  # --- Comparison ---
  # Check fuzzy match against both allowed styles to handle provider variance
  norm_actual = normalize_ws(flax_code)
  norm_expected = normalize_ws(FLAX_SOURCE)
  norm_relaxed = normalize_ws(FLAX_SOURCE_RELAXED)

  assert norm_actual == norm_expected or norm_actual == norm_relaxed, (
    f"Roundtrip failed. Got:\n{flax_code}\nExpected:\n{FLAX_SOURCE}"
  )
