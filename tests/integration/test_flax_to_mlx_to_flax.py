"""Test suite for the Flax To Mlx To Flax module."""

import pytest
import textwrap
import re
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

FLAX_SOURCE = textwrap.dedent(
  "\nfrom flax import nnx\n\nclass Net(nnx.Module):\n    def __init__(self, rngs: nnx.Rngs):\n        self.linear = nnx.Linear(10, 10, rngs=rngs)\n\n    def __call__(self, x):\n        x = self.linear(x)\n        return nnx.relu(x)\n"
).strip()


@pytest.fixture
def semantics():
  """Provides a mock semantics for testing."""
  mgr = SemanticsManager()
  mgr.data["Linear"] = {
    "std_args": ["in_features", "out_features"],
    "variants": {
      "flax_nnx": {"api": "flax.nnx.Linear"},
      "mlx": {"api": "mlx.nn.Linear", "args": {"in_features": "input_dims", "out_features": "output_dims"}},
    },
  }
  mgr._key_origins["Linear"] = SemanticTier.NEURAL.value
  mgr.data["relu"] = {
    "std_args": ["x"],
    "variants": {"flax_nnx": {"api": "flax.nnx.relu"}, "mlx": {"api": "mlx.nn.relu"}},
  }
  mgr._key_origins["relu"] = SemanticTier.ARRAY_API.value
  mgr._reverse_index["flax.nnx.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["nnx.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["flax.nnx.relu"] = ("relu", mgr.data["relu"])
  mgr._reverse_index["mlx.nn.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["nn.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._reverse_index["mlx.nn.relu"] = ("relu", mgr.data["relu"])
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
  mgr._source_registry["flax.nnx"] = ("flax_nnx", SemanticTier.NEURAL)
  mgr._source_registry["mlx.nn"] = ("mlx", SemanticTier.NEURAL)
  mgr._providers = {}
  mgr._providers["mlx"] = {SemanticTier.NEURAL: {"root": "mlx", "sub": "nn", "alias": "nn"}}
  mgr._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}
  mgr.get_all_rng_methods = lambda: set()
  mgr.framework_configs["mlx_internal"] = {"traits": {"module_base": "mlx.nn.layers.base.Module"}}
  return mgr


def normalize_ws(s):
  """Helper to normalize ws."""
  s = s.strip()
  s = re.sub("\\n+", "\n", s)
  return s


def test_flax_to_mlx_roundtrip(semantics):
  """Verifies the behavior of Flax to MLX roundtrip."""
  config_f2m = RuntimeConfig(source_framework="flax_nnx", target_framework="mlx", strict_mode=True)
  engine_f2m = ASTEngine(semantics=semantics, config=config_f2m)
  res_mlx = engine_f2m.run(FLAX_SOURCE)
  assert res_mlx.success, f"F2M Errors: {res_mlx.errors}"
  mlx_code = res_mlx.code.strip()
  assert "import mlx.nn" in mlx_code
  assert "nn.Linear" in mlx_code
  config_m2f = RuntimeConfig(source_framework="mlx", target_framework="flax_nnx", strict_mode=True)
  engine_m2f = ASTEngine(semantics=semantics, config=config_m2f)
  res_flax = engine_m2f.run(mlx_code)
  assert res_flax.success, f"M2F Errors: {res_flax.errors}"
  flax_code = res_flax.code.strip()
  assert "class Net(nnx.Module)" in flax_code
  assert "nnx.Linear" in flax_code
  assert "rngs=rngs" in flax_code
