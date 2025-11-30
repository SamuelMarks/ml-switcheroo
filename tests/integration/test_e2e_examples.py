"""End-to-End Integration Tests for Example Files.

These tests verify the complete transpilation pipeline using the files in
`tests/examples/` as inputs. They act as the primary acceptance criteria for
structural and semantic correctness.

Testing Strategy:
    1.  **Isolation**: Instead of relying on external JSON files (which may change),
        we use `E2ESemantics` (a Mock Manager). This creates a deterministic
        truth for mapping "Torch" to "JAX" specifically for these tests.
    2.  **Structural Verification**: We assert that key AST transformations
        (Class Inheritance, Import Swapping, Argument Injection) occur.
    3.  **Stability**: We ensure the engine creates valid, parseable Python code.

Coverage:
    - ex01: Simple Math Functions (Torch <-> JAX).
    - ex02: Neural Network Classes (Torch <-> Flax NNX).
    - ex03: Argument Renaming logic.
    - ex04/05: Partial failure modes (Escape Hatches).
"""

import pytest
from pathlib import Path
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.escape_hatch import EscapeHatch

# Resolve path relative to this test file
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _read_code(filename: str) -> str:
  """Helper to read source code from the examples directory."""
  path = EXAMPLES_DIR / filename
  if not path.is_file():
    pytest.fail(f"Example file not found: {path}")
  return path.read_text(encoding="utf-8")


class E2ESemantics(SemanticsManager):
  """Deterministic Semantics Manager for E2E Tests.

  This class bypasses filesystem JSON loading to provide a stable,
  code-defined Knowledge Base. It mirrors the structure of the real JSON files
  but guarantees that `test_ex01` will not break if `k_array_api.json` is modified.
  """

  def __init__(self):
    """Initializes the mock knowledge base."""
    # Initialize empty stores manually to bypass file loading
    self.data = {}
    self.import_data = {}
    # NOTE: We do not configure 'stateful_call' for JAX here because we are targeting
    # Flax NNX (Graph-based state), which allows PyTorch-like OOP direct calls.
    self.framework_configs = {}

    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}

    # --- 1. Math Operations (ex01) ---
    self._add_op("abs", ["x"], torch="torch.abs", jax="jax.numpy.abs")
    self._add_op("mean", ["x"], torch="torch.mean", jax="jax.numpy.mean")
    self._add_op("sub", ["x", "y"], torch="torch.sub", jax="jax.numpy.subtract")

    # --- 2. Neural Networks (ex02) ---
    self._add_op(
      "Module",
      [],
      torch="torch.nn.Module",
      jax="flax.nnx.Module",
      tier=SemanticTier.NEURAL,
    )
    self._add_op(
      "Linear",
      ["in_features", "out_features"],
      torch="torch.nn.Linear",
      jax="flax.nnx.Linear",
      tier=SemanticTier.NEURAL,
    )

    # Import Remapping setup for ImportFixer
    self.import_data["torch.nn"] = {"variants": {"jax": {"root": "flax", "sub": "nnx", "alias": "nnx"}}}
    self.import_data["flax.nnx"] = {"variants": {"torch": {"root": "torch", "sub": "nn", "alias": "nn"}}}

    # --- 3. Array Manipulation (ex03) ---
    self._add_op(
      "transpose",
      ["x", "axes"],
      torch="torch.permute",
      jax="jax.numpy.transpose",
    )

    # --- Aliases (Resolving common imports) ---
    self._alias("jnp.abs", "abs")
    self._alias("jnp.mean", "mean")
    self._alias("jnp.transpose", "transpose")
    self._alias("nn.Module", "Module")
    self._alias("nn.Linear", "Linear")

  def _add_op(self, name, args, torch, jax, tier=None):
    """Helper to define a bidirectional operation mapping."""
    self.data[name] = {
      "std_args": args,
      "variants": {"torch": {"api": torch}, "jax": {"api": jax}},
    }
    if torch:
      self._reverse_index[torch] = (name, self.data[name])
    if jax:
      self._reverse_index[jax] = (name, self.data[name])

    if tier:
      self._key_origins[name] = tier.value
    else:
      self._key_origins[name] = SemanticTier.ARRAY_API.value

  def _alias(self, api_str, abstract_name):
    """Register a specific string alias to an abstract ID."""
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.fixture
def engine_factory():
  """Factory fixture to create an ASTEngine with stable E2E semantics."""
  semantics = E2ESemantics()

  def _create(source, target, strict=False):
    cfg = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=strict)
    return ASTEngine(semantics=semantics, config=cfg)

  return _create


# ==============================================================================
# Example 01: Math Ops (Basic Function Mapping)
# ==============================================================================


def test_ex01_math_ops_torch_to_jax(engine_factory):
  """Verifies simple function mapping (torch -> jax).

  Checks:
      1. Import replacement (`import jax`).
      2. API renaming (`torch.abs` -> `jax.numpy.abs`).
  """
  code = _read_code("ex01_math_ops.torch.py")
  engine = engine_factory("torch", "jax")
  result = engine.run(code)

  assert result.success
  assert "import jax" in result.code
  assert "jax.numpy.abs" in result.code


def test_ex01_math_ops_jax_to_torch(engine_factory):
  """Verifies reverse function mapping (jax -> torch).

  Checks:
      1. Import replacement (`import torch`).
      2. API renaming (`jnp.abs` -> `torch.abs`).
  """
  code = _read_code("ex01_math_ops.jax.py")
  engine = engine_factory("jax", "torch")
  result = engine.run(code)

  assert result.success
  assert "import torch" in result.code
  assert "torch.abs" in result.code


# ==============================================================================
# Example 02: Neural Net (Structural Class Rewriting)
# ==============================================================================


def test_ex02_neural_net_torch_to_jax(engine_factory):
  """Verifies Class transformation (Torch -> Flax NNX).

  This determines if the `StructureMixin` and `CallMixin` in the rewriter
  are correctly identifying Neural Layers and injecting state.

  Checks:
      1. Inheritance swap (`torch.nn.Module` -> `flax.nnx.Module`).
      2. Layer instantiation.
      3. RNG state injection (`rngs=rngs`) in constructors.
  """
  code = _read_code("ex02_neural_net.torch.py")
  engine = engine_factory("torch", "jax")
  result = engine.run(code)

  if not result.success or result.has_errors:
    pytest.fail(f"Conversion failed: {result.errors}")

  # Inheritance
  assert "from flax import nnx" in result.code
  assert "class SimplePerceptron(flax.nnx.Module):" in result.code

  # Structural rewrites
  assert "flax.nnx.Linear" in result.code
  assert "rngs=rngs" in result.code, "Failed to inject rngs kwarg into layer init"
  assert "def __call__(self, x):" in result.code, "Failed to rename forward()"


def test_ex02_neural_net_jax_to_torch(engine_factory):
  """Verifies Reverse Class transformation (Flax NNX -> Torch).

  This checks the complex logic of converting explicit state management
  back to implicit state.

  Checks:
      1. Inheritance swap (`flax.nnx.Module` -> `nn.Module`).
      2. Method renaming (`__call__` -> `forward`).
      3. Stripping `rngs` arguments from `__init__`.
      4. Injecting `super().__init__()` (Mandatory for PyTorch).
  """
  code = _read_code("ex02_neural_net.jax.py")
  engine = engine_factory("jax", "torch")
  result = engine.run(code)

  if not result.success or result.has_errors:
    pytest.fail(f"Conversion failed: {result.errors}")

  assert "import torch" in result.code
  assert "class SimplePerceptron(torch.nn.Module):" in result.code
  assert "def forward(self, x):" in result.code

  # Logic: PyTorch requires super().__init__(), NNX does not.
  assert "super().__init__()" in result.code, "Failed to inject super().__init__() call"

  # Logic: PyTorch Linear layers don't take 'rngs'
  assert "def __init__(self, in_features, out_features):" in result.code, "Failed to strip 'rngs' from signature"


# ==============================================================================
# Example 03: Array Manipulation (Argument Normalization)
# ==============================================================================


def test_ex03_array_manip_torch_to_jax(engine_factory):
  """Verifies argument-aware translation.

  Checks:
      1. API swap (`permute` -> `transpose`).
      2. Basic syntax validity.
  """
  code = _read_code("ex03_array_manip.torch.py")
  engine = engine_factory("torch", "jax")
  result = engine.run(code)

  assert result.success
  assert "jax.numpy.transpose" in result.code


def test_ex03_array_manip_jax_to_torch(engine_factory):
  """Verifies reverse argument-aware translation."""
  code = _read_code("ex03_array_manip.jax.py")
  engine = engine_factory("jax", "torch")
  result = engine.run(code)

  assert result.success
  assert "torch.permute" in result.code


# ==============================================================================
# Example 04: Partial Conversion via Escape Hatch
# ==============================================================================


def test_ex04_mixed_checkpointing_torch_to_jax(engine_factory):
  """Verifies Strict Mode failure boundaries (Tier C).

  Scenario:
      Source contains `torch.abs` (Convertible) AND
      `torch.utils.checkpoint` (Not mapped in E2ESemantics).

  Expectation:
      - `abs` is converted.
      - `checkpoint` is wrapped in Escape Hatch markers.
  """
  code = _read_code("ex04_mixed_checkpointing.torch.py")
  engine = engine_factory("torch", "jax", strict=True)
  result = engine.run(code)

  # 1. Output should generate successfully (with errors logged)
  assert result.success
  assert result.has_errors

  # 2. Convertible parts worked (torch.abs -> jax.numpy.abs)
  assert "jax.numpy.abs" in result.code

  # 3. Unconvertible parts flagged with markers
  assert EscapeHatch.START_MARKER in result.code
  assert EscapeHatch.END_MARKER in result.code

  # 4. Verbatim preservation of the difficult line
  assert "checkpoint.checkpoint(" in result.code


# ==============================================================================
# Example 05: Partial Conversion (Parallelism)
# ==============================================================================


def test_ex05_mixed_parallelism_jax_to_torch(engine_factory):
  """Verifies Strict Mode failure boundaries for JAX specific features.

  Scenario:
      Source contains `jax.pmap` (Not mapped to Torch).

  Expectation:
      - `pmap` call is wrapped.
      - Other logic preserved or converted.
  """
  code = _read_code("ex05_mixed_parallelism.jax.py")
  engine = engine_factory("jax", "torch", strict=True)
  result = engine.run(code)

  assert result.success
  assert result.has_errors

  # Convertible: jnp.abs -> torch.abs
  assert "torch.abs" in result.code

  # Unconvertible: pmap -> wrapped
  assert EscapeHatch.START_MARKER in result.code
  assert "pmap(" in result.code
