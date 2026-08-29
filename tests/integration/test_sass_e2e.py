"""Test suite for the Sass E2E module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics() -> MagicMock:
  """Docstring."""
  mgr = MagicMock(spec=SemanticsManager)
  add_def: dict[str, typing.Any] = {"variants": {"torch": {"api": "torch.add"}, "sass": {"api": "FADD"}}}
  mul_def: dict[str, typing.Any] = {"variants": {"torch": {"api": "torch.mul"}, "sass": {"api": "FMUL"}}}

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    if "add" in name:
      return ("Add", add_def)
    if "mul" in name:
      return ("Mul", mul_def)
    return None

  mgr.get_definition.side_effect = get_def

  def resolve_variant(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
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
def sass_engine(semantics: MagicMock) -> ASTEngine:
  """Docstring."""
  register_framework("sass")(SassAdapter)
  config = RuntimeConfig(source_framework="torch", target_framework="sass", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


@pytest.fixture
def python_engine(semantics: MagicMock) -> ASTEngine:
  """Docstring."""
  register_framework("sass")(SassAdapter)
  config = RuntimeConfig(source_framework="sass", target_framework="jax")
  config.strict_mode = False
  return ASTEngine(semantics=semantics, config=config)


def test_python_to_sass_compilation(sass_engine: ASTEngine) -> None:
  """Verifies the behavior of python to SASS compilation."""
  source_code: str = "\nimport torch\ndef kernel(x, y):\n    z = torch.add(x, y)\n    return z\n"
  result: ConversionResult = sass_engine.run(source_code)
  assert result.success, f"Compilation failed: {result.errors}"
  output: str = result.code
  assert "// Input x -> R0" in output
  assert "// Input y -> R1" in output
  assert "FADD R2, R0, R1;" in output
  assert "// Return: R2" in output


def test_python_to_sass_unmapped_op_fallback(sass_engine: ASTEngine) -> None:
  """Verifies the behavior of python to SASS unmapped op fallback."""
  source_code: str = "z = torch.unknown(x)"
  result: ConversionResult = sass_engine.run(source_code)
  assert result.success
  output: str = result.code
  assert "// Unmapped Op:" in output
  assert "unknown" in output


def test_sass_to_python_decompilation(python_engine: ASTEngine) -> None:
  """Verifies the behavior of SASS to python decompilation."""
  sass_source: str = "FADD R0, R1, R2;"
  result: ConversionResult = python_engine.run(sass_source)
  assert result.success, f"Decompilation failed: {result.errors}"
  py_code: str = result.code
  assert "asm.FADD" in py_code or "sass.FADD" in py_code
  assert "=" in py_code


def test_full_chain_math(sass_engine: ASTEngine) -> None:
  """Verifies the behavior of full chain math."""
  source_code: str = "\nimport torch\ndef f(x, y):\n    t = torch.add(x, y)\n    return torch.mul(t, x)\n"
  result: ConversionResult = sass_engine.run(source_code)
  output: str = result.code
  assert "FADD R2, R0, R1;" in output
  assert "FMUL R3, R2, R0;" in output
