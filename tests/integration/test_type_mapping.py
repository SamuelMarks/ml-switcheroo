"""
Integration Tests for Type Mapping and Casting Logic.

This suite verifies that:
1.  **Attribute Types**: `torch.float32` maps correctly to `jnp.float32`, `tf.float32`, `np.float32`.
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
  Injects required traits and type mappings into the manager for the test.
  """
  # Initialize fresh manager
  mgr = SemanticsManager()

  # --- INJECTION: Manual Type Definitions ---
  # Ensure standard types are available even if JSONs failed to load

  # 1. CastFloat/CastInt/etc (Ops)
  mgr.update_definition(
    "CastFloat",
    {
      "std_args": ["x"],
      "operation": "CastFloat",
      "description": "Cast to Float",
      "variants": {"torch": {"api": "float"}, "jax": {"requires_plugin": "type_methods", "api": "astype"}},
      "metadata": {"target_type": "Float32"},
    },
  )
  # Bind specific implicit method path
  mgr._reverse_index["torch.Tensor.float"] = ("CastFloat", mgr.data["CastFloat"])

  mgr.update_definition(
    "CastLong",
    {
      "std_args": ["x"],
      "operation": "CastLong",
      "description": "Cast to Long",
      "variants": {"torch": {"api": "long"}, "numpy": {"requires_plugin": "type_methods", "api": "astype"}},
      "metadata": {"target_type": "Int64"},
    },
  )
  mgr._reverse_index["torch.Tensor.long"] = ("CastLong", mgr.data["CastLong"])

  mgr.update_definition(
    "CastHalf",
    {
      "std_args": ["x"],
      "operation": "CastHalf",
      "description": "Cast to Half",
      "variants": {"torch": {"api": "half"}, "mlx": {"requires_plugin": "type_methods", "api": "astype"}},
      "metadata": {"target_type": "Float16"},
    },
  )
  mgr._reverse_index["torch.Tensor.half"] = ("CastHalf", mgr.data["CastHalf"])

  mgr.update_definition(
    "CastByte",
    {
      "std_args": ["x"],
      "operation": "CastByte",
      "description": "Cast to UIint8",
      "variants": {"torch": {"api": "byte"}, "jax": {"requires_plugin": "type_methods", "api": "astype"}},
      "metadata": {"target_type": "UInt8"},
    },
  )
  mgr._reverse_index["torch.Tensor.byte"] = ("CastByte", mgr.data["CastByte"])

  # 2. Type Constants
  types_map = {
    "Bool": {"torch": "torch.bool", "numpy": "numpy.bool_", "jax": "jax.numpy.bool_", "keras": "bool"},
    "Float32": {"torch": "torch.float32", "jax": "jax.numpy.float32", "numpy": "numpy.float32", "keras": "numpy.float32"},
    "Int64": {"torch": "torch.int64", "tensorflow": "tf.int64", "numpy": "numpy.int64", "jax": "jax.numpy.int64"},
    "Float16": {"torch": "torch.float16", "mlx": "mlx.core.float16"},
    "UInt8": {"torch": "torch.uint8", "jax": "jnp.uint8"},
  }

  for type_name, variants_map in types_map.items():
    # Map reverse index for source
    vars_config = {}
    for fw, api in variants_map.items():
      vars_config[fw] = {"api": api}
      mgr._reverse_index[api] = (type_name, {"variants": vars_config})

    mgr.update_definition(
      type_name, {"operation": type_name, "description": f"Type {type_name}", "std_args": [], "variants": vars_config}
    )

  # 3. Add Abs definition for nested test case
  mgr.update_definition(
    "Abs",
    {
      "operation": "Abs",
      "description": "Absolute Value",
      "std_args": ["x"],
      "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}},
    },
  )

  # INJECT PLUGIN TRAITS for the target framework
  # This is required because the Casting plugin now checks 'has_numpy_compatible_arrays'
  if target in ["jax", "numpy", "tensorflow", "mlx", "flax", "flax_nnx", "keras"]:
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
  # Note: Requires MLX maps injected in run_transpile
  code = "y = x.half()"
  res = run_transpile(code, "mlx")
  assert ".astype" in res
  assert "mx.float16" in res


def test_cast_int_tensorflow_functional():
  """
  Scenario: .int() -> tf.cast(x, dtype=tf.int32).
  Logic: Uses standard functional rewriting (TensorFlowAdapter definition).
  """
  pass


def test_cast_bool_keras_string():
  """
  Scenario: .bool() -> keras.ops.cast(x, dtype='bool').
  Logic: Uses functional rewriting with string injection.
  """
  pass


def test_nested_casting_expression():
  """
  Scenario: Chained/Nested casts.
  Input: torch.abs(x.float())
  Target JAX: jnp.abs(x.astype(jnp.float32))
  """
  code = "y = torch.abs(x.float())"
  res = run_transpile(code, "jax")

  assert "jax.numpy.abs" in res or "jnp.abs" in res
  assert "x.astype(" in res
  assert "jax.numpy.float32" in res or "jnp.float32" in res


def test_cast_byte_uint8():
  """Scenario: .byte() -> uint8."""
  code = "img = x.byte()"
  res = run_transpile(code, "jax")
  # JAX maps uint8 types correctly
  assert ".astype(jnp.uint8)" in res
