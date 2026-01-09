"""
Integration Tests for Type Mapping and Casting Logic.
"""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits
from ml_switcheroo.enums import SemanticTier


# Ensure plugins are loaded
@pytest.fixture(autouse=True)
def reload_plugins():
  from ml_switcheroo.core import hooks
  import ml_switcheroo.plugins.casting

  hooks._PLUGINS_LOADED = False
  importlib.reload(ml_switcheroo.plugins.casting)
  hooks.load_plugins()


def run_transpile(code: str, target: str) -> str:
  mgr = SemanticsManager()

  # Manual Injection for testing
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

  # Providers for Import Injection
  mgr._providers = {}
  mgr._providers["keras"] = {SemanticTier.ARRAY_API: {"root": "numpy", "sub": None, "alias": "np"}}

  # Source Registry setup (mocking origin)
  mgr._source_registry["torch.float32"] = ("torch", SemanticTier.ARRAY_API)
  mgr._key_origins["Float32"] = SemanticTier.ARRAY_API.value

  # Traits
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
  """
  Verify torch.float32 -> np.float32.
  Expect: 'import numpy as np' injected because Keras relies on numpy types.
  """
  code = "dtype = torch.float32"
  res = run_transpile(code, "keras")

  assert "import numpy as np" in res
  assert "np.float32" in res
