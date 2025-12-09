import pytest
import ast
from pathlib import Path
from typing import Set

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

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
  """

  def __init__(self):
    # Bypass filesystem load
    self.data = {}
    self.import_data = {}
    self.framework_configs = {"jax": {"stateful_call": {"method": "apply", "prepend_arg": "variables"}}}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # 1. Bidirectional Math Mappings
    self._add_op("abs", ["x"], torch="torch.abs", jax="jax.numpy.abs")
    self._add_op("mean", ["x"], torch="torch.mean", jax="jax.numpy.mean")

    # 2. Bidirectional Array Mappings
    # torch.permute <-> jax.numpy.transpose
    # Note: In real scenarios, args mapping differs (dims vs axes), simplified here for demo
    self._add_op("transpose", ["x", "axes"], torch="torch.permute", jax="jax.numpy.transpose")

    # 3. Neural Layers
    self._add_op("Linear", ["in", "out"], torch="torch.nn.Linear", jax="flax.nnx.Linear")
    self._add_op("Module", [], torch="torch.nn.Module", jax="flax.nnx.Module")

    # 4. Register Aliases (Idiomatic usage support)
    # JAX aliases
    self._alias("jnp.abs", "abs")
    self._alias("jnp.mean", "mean")
    self._alias("jnp.transpose", "transpose")

    # Torch aliases
    self._alias("nn.Linear", "Linear")
    self._alias("nn.Module", "Module")

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def _add_op(self, name, args, torch, jax):
    """Helper to define a bidirectional op."""
    self.data[name] = {"std_args": args, "variants": {"torch": {"api": torch}, "jax": {"api": jax}}}
    # Register canonical reverse lookups
    self._alias(torch, name)
    self._alias(jax, name)

  def _alias(self, api_str, abstract_name):
    """Map a specific string string to an abstract definition."""
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.mark.parametrize("example_file", _load_files())
def test_transpile_generates_valid_code(example_file):
  """
  Core Compatibility Test.

  Verifies that:
  1. .torch.py files convert to JAX.
  2. .jax.py files convert to Torch.
  3. Output is syntactically valid Python.
  4. Target libraries are imported.
  """
  filename = example_file.name

  if filename.endswith(".torch.py"):
    source_fw, target_fw = "torch", "jax"
    target_indicators = ["jax", "jnp", "flax"]
  elif filename.endswith(".jax.py"):
    # Explicit check for JAX -> Torch direction
    source_fw, target_fw = "jax", "torch"
    target_indicators = ["torch", "nn"]

    # Skip Class-based JAX->Torch reverse translation if not fully implemented in Rewriter
    # (NNX classes have different init signatures that require complex un-rewriting)
    if "neural_net" in filename:
      pytest.skip("Reversing NNX classes to Torch classes is WIP.")
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
  # (Checking if ANY of the target indicators appear in the output code)
  has_target = any(ind in result.code for ind in target_indicators)

  if not has_target:
    pytest.fail(
      f"Output does not contain expected {target_fw} imports/modules.\n"
      f"Expected one of: {target_indicators}\n"
      f"Generated:\n{result.code}"
    )

  print("✅ Verified.")
