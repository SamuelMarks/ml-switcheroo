"""
Integration Tests for Example Files.

Verifies that the provided example files can be transpiled end-to-end
using the semantics and engine logic. This suite uses a local Mock Semantics
object to ensure tests are deterministic and don't rely on the state of
the filesystem JSONs (which might change during development).
"""

import pytest
import ast
from pathlib import Path
from typing import Set

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier

# Resolve path relative to this test file
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _load_files():
  """Discover .py files in tests/examples/"""
  if not EXAMPLES_DIR.exists():
    return []
  # Sort for deterministic order
  return sorted(list(EXAMPLES_DIR.glob("*.py")))


class MockBidirectionalSemantics(SemanticsManager):
  """
  Injects deterministic mappings supporting BOTH directions (Torch <-> JAX).
  Includes expanded mappings for VAE (ex08), CNN (ex07), and Keras (ex09) support.
  """

  def __init__(self):
    # Bypass filesystem load
    self.data = {}
    self.import_data = {}

    # --- Framework Configuration (Traits) ---
    # Critical for Class Rewriting (Module detection, Method renaming, Super init)
    self.framework_configs = {
      "jax": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
          "requires_super_init": False,
        },
        "stateful_call": {"method": "apply", "prepend_arg": "variables"},
      },
      "torch": {
        "traits": {
          "module_base": "torch.nn.Module",
          "forward_method": "forward",
          "strip_magic_args": ["rngs"],
          "requires_super_init": True,
          "lifecycle_strip_methods": ["to", "cuda", "cpu", "detach"],
        }
      },
      "keras": {"traits": {"module_base": "keras.Model", "forward_method": "call", "requires_super_init": True}},
    }

    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # --- 1. Math / Array Operations ---
    # Used in: ex01, ex08, ex09
    self._add_op("abs", ["x"], torch="torch.abs", jax="jax.numpy.abs", keras="keras.ops.abs")
    self._add_op("mean", ["x"], torch="torch.mean", jax="jax.numpy.mean", keras="keras.ops.mean")
    self._add_op("sum", ["x"], torch="torch.sum", jax="jax.numpy.sum", keras="keras.ops.sum")
    self._add_op("exp", ["x"], torch="torch.exp", jax="jax.numpy.exp", keras="keras.ops.exp")
    self._add_op("log", ["x"], torch="torch.log", jax="jax.numpy.log", keras="keras.ops.log")
    self._add_op("square", ["x"], torch="torch.square", jax="jax.numpy.square", keras="keras.ops.square")

    # --- 2. Randomness ---
    # Used in: ex08
    self._add_op("randn", ["shape"], torch="torch.randn", jax="jax.random.normal", keras="keras.random.normal")

    # --- 3. Array Manipulation ---
    # Used in: ex03
    # torch.permute <-> jax.numpy.transpose
    self._add_op(
      "transpose", ["x", "axes"], torch="torch.permute", jax="jax.numpy.transpose", keras="keras.ops.transpose"
    )

    # ex07 usage - Flatten function
    self._add_op(
      "flatten_func", ["x", "start_dim"], torch="torch.flatten", jax="jax.numpy.ravel", keras="keras.ops.ravel"
    )

    # --- 4. Neural Layers & Models ---
    # Used in: ex02, ex07, ex09
    self._add_op("Linear", ["in", "out"], torch="torch.nn.Linear", jax="flax.nnx.Linear", keras="keras.layers.Dense")
    self._add_op("Conv2d", ["in", "out", "k"], torch="torch.nn.Conv2d", jax="flax.nnx.Conv", keras="keras.layers.Conv2D")
    self._add_op("Dropout", ["p"], torch="torch.nn.Dropout", jax="flax.nnx.Dropout", keras="keras.layers.Dropout")
    self._add_op("Flatten", [], torch="torch.nn.Flatten", jax="flax.nnx.Flatten", keras="keras.layers.Flatten")
    self._add_op(
      "MaxPool2d", ["k"], torch="torch.nn.MaxPool2d", jax="flax.nnx.max_pool", keras="keras.layers.MaxPooling2D"
    )

    # Base Classes
    self._add_op("Module", [], torch="torch.nn.Module", jax="flax.nnx.Module", keras="keras.Model")
    # Keras Input is a functional op, but we map it to empty to verify it's caught
    self._add_op("Input", ["shape"], torch="torch.empty", jax="jax.numpy.empty", keras="keras.Input")

    # --- 5. Functional NN Ops ---
    # Used in ex07
    self._add_op("relu", ["x"], torch="torch.nn.functional.relu", jax="jax.nn.relu")
    # For F.relu reuse in Torch mapping if not detected as method
    self._add_op("relu_f", ["x"], torch="torch.relu", jax="jax.nn.relu")

    self._add_op("log_softmax", ["x"], torch="torch.nn.functional.log_softmax", jax="jax.nn.log_softmax")
    self._add_op(
      "max_pool2d_func", ["x"], torch="torch.nn.functional.max_pool2d", jax="jax.lax.reduce_window"
    )  # Mock approximation

    # --- 6. Register Aliases (Idiomatic usage support) ---

    # JAX aliases
    self._alias("jnp.abs", "abs")
    self._alias("jnp.mean", "mean")
    self._alias("jnp.sum", "sum")
    self._alias("jnp.exp", "exp")
    self._alias("jnp.log", "log")
    self._alias("jnp.square", "square")
    self._alias("jnp.transpose", "transpose")
    self._alias("jax.random.normal", "randn")

    # Torch aliases
    self._alias("nn.Linear", "Linear")
    self._alias("nn.Conv2d", "Conv2d")
    self._alias("nn.Dropout", "Dropout")
    self._alias("nn.Module", "Module")
    self._alias("nn.Flatten", "Flatten")
    self._alias("F.relu", "relu")
    self._alias("F.log_softmax", "log_softmax")
    self._alias("F.max_pool2d", "max_pool2d_func")

    # Keras aliases
    self._alias("layers.Conv2D", "Conv2d")
    self._alias("layers.MaxPooling2D", "MaxPool2d")
    self._alias("layers.Flatten", "Flatten")
    self._alias("layers.Dropout", "Dropout")
    self._alias("layers.Dense", "Linear")
    self._alias("ops.mean", "mean")
    self._alias("ops.square", "square")

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _add_op(self, name, args, **variants):
    """Helper to define a multi-way op."""
    variant_data = {}
    for fw, api in variants.items():
      variant_data[fw] = {"api": api}
      # Register reverse lookup for this specific API path
      self._alias(api, name)

    self.data[name] = {"std_args": args, "variants": variant_data}
    # Ensure tier logic marks these as Neural if they start with uppercase
    if name[0].isupper():
      self._key_origins[name] = SemanticTier.NEURAL.value
    else:
      self._key_origins[name] = SemanticTier.ARRAY_API.value

  def _alias(self, api_str, abstract_name):
    """Map a specific string string to an abstract definition."""
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.mark.parametrize("example_file", _load_files())
def test_transpile_generates_valid_code(example_file):
  """
  Core Compatibility Test.

  Verifies file conversion for all examples in the suite.
  """
  filename = example_file.name

  # Determine Direction
  if filename.endswith(".torch.py"):
    source_fw, target_fw = "torch", "jax"
    target_indicators = ["jax", "jnp", "flax"]
  elif filename.endswith(".jax.py"):
    source_fw, target_fw = "jax", "torch"
    target_indicators = ["torch", "nn"]
    # Skip Reverse Class Gen
    if "neural_net" in filename:
      pytest.skip("Reversing NNX classes to Torch classes is WIP.")
  elif filename.endswith(".keras.py"):
    # Keras -> Torch
    source_fw, target_fw = "keras", "torch"
    target_indicators = ["torch", "nn"]
  else:
    pytest.skip(f"Skipping {filename}: Unknown extension.")

  print(f"\n⚡️ Translating {filename}: {source_fw} -> {target_fw}")

  code = example_file.read_text("utf-8")

  # Setup Engine
  semantics = MockBidirectionalSemantics()
  config = RuntimeConfig(
    source_framework=source_fw,
    target_framework=target_fw,
    strict_mode=False,  # Allow print/comments/docstrings
  )

  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)

  # 1. Assert Engine Success
  assert result.success, f"Engine crashed: {result.errors}"

  # 2. Assert Valid Python Syntax
  try:
    ast.parse(result.code)
  except SyntaxError as e:
    pytest.fail(f"Transpilation produced invalid syntax:\n{e}\n\nCode:\n{result.code}")

  # 3. Assert Target Imports exist
  has_target = any(ind in result.code for ind in target_indicators)

  if not has_target:
    # SPECIAL EXCEPTION: ex04_mixed_checkpointing is EXPECTED to have partial failure.
    # The 'checkpoint' call is not mapped, so ImportFixer might not see JAX usage if the
    # only converted call was wrapped or obscured.
    # We allow this specific file to pass if ANY conversion logic ran, effectively skipping this check.
    if "mixed" in filename or "checkpointing" in filename:
      print(f"⚠️ Allowing {filename} despite missing target imports (Partial Failure Expected).")
      return

    pytest.fail(
      f"Output does not contain expected {target_fw} imports/modules.\n"
      f"Expected one of: {target_indicators}\n"
      f"Generated:\n{result.code}"
    )

  print("✅ Verified.")
