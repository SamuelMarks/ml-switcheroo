"""Test suite for the Sass E2E module."""

from pydantic import JsonValue
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.frameworks.nvidia_sass import NvidiaSassAdapter
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics() -> MagicMock:
  """Execute semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr = MagicMock(spec=SemanticsManager)
  add_def = {"variants": {"torch": {"api": "torch.add"}, "nvidia_sass": {"api": "FADD"}}}
  mul_def = {"variants": {"torch": {"api": "torch.mul"}, "nvidia_sass": {"api": "FMUL"}}}

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "add" in name:
      _ret_v_1: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": (getattr({"v": "Add"}, "__getitem__")("v"), getattr({"v": add_def}, "__getitem__")("v"))}, "__getitem__"
      )("v")
      return _ret_v_1
    if "mul" in name:
      _ret_v_2: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": (getattr({"v": "Mul"}, "__getitem__")("v"), getattr({"v": mul_def}, "__getitem__")("v"))}, "__getitem__"
      )("v")
      return _ret_v_2
    return None

  mgr.get_definition.side_effect = get_def

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "Add" and fw == "nvidia_sass":
      return {"api": "FADD"}
    if aid == "Mul" and fw == "nvidia_sass":
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
  """Execute sass engine helper logic.

  Args:
      semantics: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  register_framework("nvidia_sass")(NvidiaSassAdapter)
  config = RuntimeConfig(source_framework="torch", target_framework="nvidia_sass", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


@pytest.fixture
def python_engine(semantics: MagicMock) -> ASTEngine:
  """Execute python engine helper logic.

  Args:
      semantics: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  register_framework("nvidia_sass")(NvidiaSassAdapter)
  config = RuntimeConfig(source_framework="nvidia_sass", target_framework="jax")
  config.strict_mode = False
  return ASTEngine(semantics=semantics, config=config)


def test_python_to_sass_compilation(sass_engine: ASTEngine) -> None:
  """Verifies the behavior of python to NVIDIA_SASS compilation.

  Args:
      sass_engine: Test argument or fixture.
  """
  source_code: str = "\nimport torch\ndef kernel(x, y):\n    z = torch.add(x, y)\n    return z\n"
  result: ConversionResult = sass_engine.run(source_code)
  assert result.success, f"Compilation failed: {result.errors}"
  output: str = result.code
  assert "// Input x -> R0" in output
  assert "// Input y -> R1" in output
  assert "fadd" in output.lower()
  assert "// Return: R2" in output


def test_python_to_sass_unmapped_op_fallback(sass_engine: ASTEngine) -> None:
  """Verifies the behavior of python to NVIDIA_SASS unmapped op fallback.

  Args:
      sass_engine: Test argument or fixture.
  """
  source_code: str = "z = torch.unknown(x)"
  result: ConversionResult = sass_engine.run(source_code)
  assert result.success
  output: str = result.code
  assert "// Unmapped Op:" in output
  assert "unknown" in output


def test_nvidia_sass_to_python_decompilation(python_engine: ASTEngine) -> None:
  """Verifies the behavior of NVIDIA_SASS to python decompilation.

  Args:
      python_engine: Test argument or fixture.
  """
  sass_source: str = "FADD R0, R1, R2;"
  result: ConversionResult = python_engine.run(sass_source)
  assert result.success, f"Decompilation failed: {result.errors}"
  py_code: str = result.code
  assert "asm.FADD" in py_code or "nvidia_sass.FADD" in py_code
  assert "=" in py_code


def test_full_chain_math(sass_engine: ASTEngine) -> None:
  """Verifies the behavior of full chain math.

  Args:
      sass_engine: Test argument or fixture.
  """
  source_code: str = "\nimport torch\ndef f(x, y):\n    t = torch.add(x, y)\n    return torch.mul(t, x)\n"
  result: ConversionResult = sass_engine.run(source_code)
  output: str = result.code
  assert "fadd" in output.lower()
  assert "fmul" in output.lower()
