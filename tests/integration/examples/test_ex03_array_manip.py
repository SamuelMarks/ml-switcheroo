"""Test suite for the Ex03 Array Manip module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier

SOURCE_TORCH = "\nimport torch\n\ndef transpose_matrices(batch):\n    return torch.permute(batch, 0, 2, 1)\n"


@pytest.fixture(scope="module")
def semantics():
  """Helper to semantics."""
  mgr = SemanticsManager()
  mgr._providers["numpy"] = {SemanticTier.ARRAY_API: {"root": "numpy", "sub": None, "alias": "np"}}
  mgr.framework_configs["numpy"] = {"alias": {"module": "numpy", "name": "np"}}
  op_data = {
    "operation": "permute_dims",
    "description": "Permute tensor dimensions.",
    "std_args": ["x", {"name": "axes", "is_variadic": True}],
    "variants": {
      "jax": {"api": "jnp.transpose", "pack_to_tuple": "axes"},
      "tensorflow": {"api": "tf.transpose", "pack_to_tuple": "perm"},
      "numpy": {"api": "numpy.transpose", "pack_to_tuple": "axes"},
    },
  }
  mgr.update_definition("permute_dims", op_data)
  mgr._reverse_index["torch.permute"] = ("permute_dims", mgr.data["permute_dims"])
  mgr._key_origins["permute_dims"] = SemanticTier.ARRAY_API.value
  return mgr


@pytest.mark.parametrize(
  "target_fw, structural_check",
  [
    ("jax", "jnp.transpose(batch, axes=(0, 2, 1))"),
    ("tensorflow", "tf.transpose(batch, perm=(0, 2, 1))"),
    ("numpy", "np.transpose(batch, axes=(0, 2, 1))"),
  ],
)
def test_ex03_permute_plugin(semantics, target_fw, structural_check):
  """Verifies the behavior of ex03 permute plugin."""
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_TORCH)
  assert result.success, f"Errors: {result.errors}"
  assert structural_check in result.code
