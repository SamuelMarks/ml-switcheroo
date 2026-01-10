"""
End-to-End Integration Tests for SASS Compilation.

Verifies the full pipeline through the ASTEngine:
1.  **Python -> SASS**: Converts Python logic to Assembly text via Graph synthesis and Register allocation.
2.  **SASS -> Python**: Ingests Assembly text and produces a Python AST representation.
3.  **Roundtrip**: Verify SASS->Python generated code is valid.
"""

import pytest
from unittest.mock import MagicMock

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.frameworks import register_framework

# --- Fixtures ---


@pytest.fixture
def semantics():
  """
  Mock SemanticsManager with SASS definitions to avoid relying on external JSON files.
  """
  mgr = MagicMock(spec=SemanticsManager)

  # 1. Define 'Add' -> 'FADD'
  add_def = {"variants": {"torch": {"api": "torch.add"}, "sass": {"api": "FADD"}}}

  # 2. Define 'Mul' -> 'FMUL'
  mul_def = {"variants": {"torch": {"api": "torch.mul"}, "sass": {"api": "FMUL"}}}

  # 3. Lookup Logic
  # Reverse Index Mock
  def get_def(name):
    if "add" in name:
      return "Add", add_def
    if "mul" in name:
      return "Mul", mul_def
    return None

  mgr.get_definition.side_effect = get_def

  # Resolver Logic
  def resolve_variant(aid, fw):
    if aid == "Add" and fw == "sass":
      return {"api": "FADD"}
    if aid == "Mul" and fw == "sass":
      return {"api": "FMUL"}
    return None

  mgr.resolve_variant.side_effect = resolve_variant

  # Safe Defaults
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}
  mgr.get_framework_aliases.return_value = {}

  # For Python Frontend
  mgr.get_all_rng_methods.return_value = set()

  return mgr


@pytest.fixture
def sass_engine(semantics):
  """Engine configured for Torch -> SASS."""
  # Ensure SASS adapter is registered (it is by default, but fixture ensures safety)
  register_framework("sass")(SassAdapter)

  config = RuntimeConfig(source_framework="torch", target_framework="sass", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


@pytest.fixture
def python_engine(semantics):
  """Engine configured for SASS -> JAX (Python Source)."""
  register_framework("sass")(SassAdapter)
  config = RuntimeConfig(source_framework="sass", target_framework="jax")
  # Enable strict mode false to avoid linter complaining about 'asm' imports failing
  config.strict_mode = False
  return ASTEngine(semantics=semantics, config=config)


# --- Tests ---


def test_python_to_sass_compilation(sass_engine):
  """
  Scenario: Convert Python logic `z = torch.add(x, y)` to SASS.
  Expectation:
  - Input comments.
  - FADD instruction with registers.
  """
  source_code = """ 
import torch
def kernel(x, y): 
    z = torch.add(x, y) 
    return z
"""
  result = sass_engine.run(source_code)

  assert result.success, f"Compilation failed: {result.errors}"
  output = result.code

  # 1. Inputs should be commented
  assert "// Input x -> R0" in output
  assert "// Input y -> R1" in output

  # 2. Add Op logic
  # Expect FADD (from semantics)
  # R2 = R0 + R1 usually, but allocator linear
  assert "FADD R2, R0, R1;" in output

  # 3. Output
  assert "// Return: R2" in output


def test_python_to_sass_unmapped_op_fallback(sass_engine):
  """
  Scenario: Op without SASS definition (e.g. unknown).
  Expectation: Comment fallback `// Unmapped Op: ...`
  """
  source_code = "z = torch.unknown(x)"

  result = sass_engine.run(source_code)
  assert result.success
  output = result.code

  assert "// Unmapped Op:" in output
  assert "unknown" in output


def test_sass_to_python_decompilation(python_engine):
  """
  Scenario: Convert SASS source `FADD R0, R1, R2;` to Python representation.
  Expectation: Class DecompiledModel with `asm.FADD` calls.
  """
  sass_source = "FADD R0, R1, R2;"

  result = python_engine.run(sass_source)

  assert result.success, f"Decompilation failed: {result.errors}"
  py_code = result.code

  # Check that the functional call is present
  # The Python Backend assigns "x = ..." to chain dataflow from the input.
  # It does NOT preserve register IDs as variable names for assignment, reusing 'x' (current_var).
  assert "asm.FADD" in py_code or "sass.FADD" in py_code

  # We assert assignment happens, even if variable name is 'x'
  assert "=" in py_code


def test_full_chain_math(sass_engine):
  """
  Scenario: Chained operations. `z = (x + y) * x`
  """
  source_code = """ 
import torch
def f(x, y): 
    t = torch.add(x, y) 
    return torch.mul(t, x) 
"""
  result = sass_engine.run(source_code)
  output = result.code

  assert "FADD R2, R0, R1;" in output
  assert "FMUL R3, R2, R0;" in output
