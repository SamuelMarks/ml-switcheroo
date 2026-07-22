"""Test suite for the Sass E2E module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.frameworks import register_framework


@pytest.fixture
def semantics():
  """Provides a mock semantics for testing."""
  mgr = MagicMock(spec=SemanticsManager)
  add_def = {"variants": {"torch": {"api": "torch.add"}, "sass": {"api": "FADD"}}}
  mul_def = {"variants": {"torch": {"api": "torch.mul"}, "sass": {"api": "FMUL"}}}

  def get_def(name):
    """Gets def."""
    if "add" in name:
      return ("Add", add_def)
    if "mul" in name:
      return ("Mul", mul_def)
    return None

  mgr.get_definition.side_effect = get_def

  def resolve_variant(aid, fw):
    """Resolves variant."""
    if aid == "Add" and fw == "sass":
      return {"api": "FADD"}
    if aid == "Mul" and fw == "sass":
      return {"api": "FMUL"}
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}
  mgr.get_framework_aliases.return_value = {}
  mgr.get_all_rng_methods.return_value = set()
  return mgr


@pytest.fixture
def sass_engine(semantics):
  """Provides a mock SASS engine for testing."""
  register_framework("sass")(SassAdapter)
  config = RuntimeConfig(source_framework="torch", target_framework="sass", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


@pytest.fixture
def python_engine(semantics):
  """Provides a mock python engine for testing."""
  register_framework("sass")(SassAdapter)
  config = RuntimeConfig(source_framework="sass", target_framework="jax")
  config.strict_mode = False
  return ASTEngine(semantics=semantics, config=config)


def test_python_to_sass_compilation(sass_engine):
  """Verifies the behavior of python to SASS compilation."""
  source_code = "\nimport torch\ndef kernel(x, y):\n    z = torch.add(x, y)\n    return z\n"
  result = sass_engine.run(source_code)
  assert result.success, f"Compilation failed: {result.errors}"
  output = result.code
  assert "// Input x -> R0" in output
  assert "// Input y -> R1" in output
  assert "FADD R2, R0, R1;" in output
  assert "// Return: R2" in output


def test_python_to_sass_unmapped_op_fallback(sass_engine):
  """Verifies the behavior of python to SASS unmapped op fallback."""
  source_code = "z = torch.unknown(x)"
  result = sass_engine.run(source_code)
  assert result.success
  output = result.code
  assert "// Unmapped Op:" in output
  assert "unknown" in output


def test_sass_to_python_decompilation(python_engine):
  """Verifies the behavior of SASS to python decompilation."""
  sass_source = "FADD R0, R1, R2;"
  result = python_engine.run(sass_source)
  assert result.success, f"Decompilation failed: {result.errors}"
  py_code = result.code
  assert "asm.FADD" in py_code or "sass.FADD" in py_code
  assert "=" in py_code


def test_full_chain_math(sass_engine):
  """Verifies the behavior of full chain math."""
  source_code = "\nimport torch\ndef f(x, y):\n    t = torch.add(x, y)\n    return torch.mul(t, x)\n"
  result = sass_engine.run(source_code)
  output = result.code
  assert "FADD R2, R0, R1;" in output
  assert "FMUL R3, R2, R0;" in output
