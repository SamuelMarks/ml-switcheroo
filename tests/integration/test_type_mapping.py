"""
Integration Tests for Type Mapping and Casting Logic.

This suite verifies that:
1.  **Attribute Types**: `torch.float32` maps correctly to `jnp.float32`, `tf.float32`, `np.float32` (Keras).
2.  **Method Casting**: Method shorthands (`.float()`, `.long()`) are transformed into the
    target framework's specific idiom using `PluginTraits` logic.
"""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits


# Ensure plugins are loaded for this test session by forcing reload
@pytest.fixture(autouse=True)
def reload_plugins():
  from ml_switcheroo.core import hooks
  import ml_switcheroo.plugins.casting

  # Reset internal flag to allow load logic
  hooks._PLUGINS_LOADED = False
  # Reload specific plugin module to re-trigger @register_hook
  importlib.reload(ml_switcheroo.plugins.casting)
  hooks.load_plugins()


def run_transpile(code: str, target: str) -> str:
  """
  Helper to run the engine end-to-end for a specific target.
  Injects required traits into the manager for the test target.
  """
  # Initialize fresh manager
  mgr = SemanticsManager()

  # Ensure CastFloat/CastInt definitions exist in manager
  if "CastFloat" not in mgr.get_known_apis():
    pytest.fail("SemanticsManager did not load CastFloat standards. Check standards_internal.py.")

  # INJECT PLUGIN TRAITS for the target framework
  # This is required because the Casting plugin now checks 'has_numpy_compatible_arrays'
  if target in ["jax", "numpy", "tensorflow", "mlx", "flax", "flax_nnx"]:
    if target not in mgr.framework_configs:
      mgr.framework_configs[target] = {}

    # Enable numpy compat
    mgr.framework_configs[target]["plugin_traits"] = PluginTraits(has_numpy_compatible_arrays=True)

  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  engine = ASTEngine(semantics=mgr, config=cfg)
  result = engine.run(code)

  if not result.success:
    print(f"Error Code: {result.code}")
    pytest.fail(f"Transpilation failed for {target}: {result.errors}")

  return result.code


# --- 1. Attribute Type Mapping Tests ---


def test_type_constant_jax():
  """Verify torch.float32 -> jnp.float32"""
  code = "dtype = torch.float32"
  res = run_transpile(code, "jax")
  assert "jnp.float32" in res


def test_type_constant_tensorflow():
  """Verify torch.int64 -> tf.int64"""
  code = "dtype = torch.int64"
  res = run_transpile(code, "tensorflow")
  assert "tf.int64" in res


def test_type_constant_mlx():
  """Verify torch.float16 -> mx.float16"""
  code = "dtype = torch.float16"
  res = run_transpile(code, "mlx")
  assert "mx.float16" in res


def test_type_constant_numpy():
  """Verify torch.bool -> np.bool_"""
  code = "dtype = torch.bool"
  res = run_transpile(code, "numpy")
  assert "np.bool_" in res


def test_type_constant_keras():
  """
  Verify torch.float32 -> np.float32 (Keras uses numpy types for objects via 'np' alias).
  """
  code = "dtype = torch.float32"
  res = run_transpile(code, "keras")
  # ImportFixer injects 'import numpy as np' because definitions request it
  assert "import numpy as np" in res
  assert "np.float32" in res


# --- 2. Casting Logic Tests (Plugin vs Functional) ---


def test_cast_float_jax_plugin():
  """
  Scenario: .float() -> .astype(jnp.float32) via 'type_methods' plugin.
  Method name lookup 'float' -> 'CastFloat' required in reverse index.
  """
  code = "y = x.float()"
  res = run_transpile(code, "jax")
  assert ".astype" in res
  assert "jnp.float32" in res
  assert ".float(" not in res


def test_cast_long_numpy_plugin():
  """Scenario: .long() -> .astype(np.int64)."""
  code = "y = x.long()"
  res = run_transpile(code, "numpy")
  assert ".astype" in res
  assert "np.int64" in res


def test_cast_half_mlx_plugin():
  """Scenario: .half() -> .astype(mx.float16)."""
  code = "y = x.half()"
  res = run_transpile(code, "mlx")
  assert ".astype" in res
  assert "mx.float16" in res


def test_cast_int_tensorflow_functional():
  """
  Scenario: .int() -> tf.cast(x, dtype=tf.int32).
  Logic: Uses standard functional rewriting (TensorFlowAdapter definition).
  """
  code = "y = x.int()"
  res = run_transpile(code, "tensorflow")
  assert "tf.cast" in res
  assert "dtype='tf.int32'" in res or 'dtype="tf.int32"' in res
  assert "tf.cast(x" in res


def test_cast_bool_keras_string():
  """
  Scenario: .bool() -> keras.ops.cast(x, dtype='bool').
  Logic: Uses functional rewriting with string injection.
  """
  code = "mask = x.bool()"
  res = run_transpile(code, "keras")
  assert "keras.ops.cast" in res
  assert 'dtype="bool"' in res or "dtype='bool'" in res


def test_nested_casting_expression():
  """
  Scenario: Chained/Nested casts.
  Input: torch.abs(x.float())
  Target JAX: jnp.abs(x.astype(jnp.float32))
  """
  code = "y = torch.abs(x.float())"
  res = run_transpile(code, "jax")
  assert "jnp.abs" in res
  assert "x.astype(" in res
  assert "jnp.float32" in res


def test_cast_byte_uint8():
  """Scenario: .byte() -> uint8."""
  code = "img = x.byte()"
  res = run_transpile(code, "jax")
  # JAX maps uint8 types correctly
  assert ".astype(jnp.uint8)" in res
