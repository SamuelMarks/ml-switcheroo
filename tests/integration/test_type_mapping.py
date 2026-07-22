"""Test suite for the Type Mapping module."""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits
from ml_switcheroo_ir.schema.ghost import SemanticTier


@pytest.fixture(autouse=True)
def reload_plugins():
  """Helper to reload plugins."""
  from ml_switcheroo.core import hooks
  import ml_switcheroo.plugins.casting

  hooks._PLUGINS_LOADED = False
  importlib.reload(ml_switcheroo.plugins.casting)
  hooks.load_plugins()


def run_transpile(code: str, target: str) -> str:
  """Runs transpile."""
  mgr = SemanticsManager()
  mgr.update_definition(
    "CastFloat",
    {
      "variants": {"torch": {"api": "float"}, "jax": {"api": "astype", "requires_plugin": "type_methods"}},
      "metadata": {"target_type": "Float32"},
      "std_args": ["x"],
    },
  )
  mgr._reverse_index["torch.Tensor.float"] = ("CastFloat", mgr.data["CastFloat"])
  mgr.update_definition(
    "Float32",
    {
      "variants": {
        "jax": {"api": "jax.numpy.float32"},
        "numpy": {"api": "numpy.float32"},
        "keras": {"api": "numpy.float32"},
      }
    },
  )
  mgr._reverse_index["torch.float32"] = ("Float32", mgr.data["Float32"])
  mgr._providers = {}
  mgr._providers["keras"] = {SemanticTier.ARRAY_API: {"root": "numpy", "sub": None, "alias": "np"}}
  mgr._source_registry["torch.float32"] = ("torch", SemanticTier.ARRAY_API)
  mgr._key_origins["Float32"] = SemanticTier.ARRAY_API.value
  if target not in mgr.framework_configs:
    mgr.framework_configs[target] = {}
  mgr.framework_configs[target]["plugin_traits"] = PluginTraits(has_numpy_compatible_arrays=True)
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  engine = ASTEngine(semantics=mgr, config=cfg)
  res = engine.run(code)
  if not res.success:
    pytest.fail(str(res.errors))
  return res.code


def test_type_constant_keras():
  """Verifies the behavior of type constant Keras."""
  code = "dtype = torch.float32"
  res = run_transpile(code, "keras")
  assert "import numpy as np" in res
  assert "np.float32" in res
