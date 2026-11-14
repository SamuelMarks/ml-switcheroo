"""Test suite for the Engine Switching module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.semantics.manager import SemanticsManager

register_framework("mlir")(MlirAdapter)


@pytest.fixture
def base_engine() -> typing.Callable[[str, str], ASTEngine]:
  """Create a configured `base_engine` fixture for engine switching tests.

  Returns:
      Configured `base_engine` fixture value (typing.Callable[[str, str], ASTEngine]).
  """
  semantics = MagicMock(spec=SemanticsManager)
  semantics.get_framework_config.return_value = {}
  semantics.get_import_map.return_value = {}
  semantics.get_framework_aliases.return_value = {}

  def get_def_side_effect(name: str) -> None:
    """Gets def side effect.

    Args:
        name: Identifier or operation name string.
    """
    return

  semantics.get_definition.side_effect = get_def_side_effect

  def create(source: str, target: str) -> ASTEngine:
    """Execute `RuntimeConfig` to support create in engine switching tests.

    Args:
        source: Source framework identifier or input code string.
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `ASTEngine` from create.
    """
    config = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=False, enable_import_fixer=False)
    config.validation_report = None
    return ASTEngine(semantics, config)

  return create


def test_python_to_mlir(base_engine: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verify `base_engine` emits `sw.constant` for python to MLIR.

  Args:
      base_engine: Input `base_engine` parameter value (`typing.Callable[[str, str], ASTEngine]`).
  """
  engine = base_engine("torch", "mlir")
  code: str = "x = 1"
  result: ConversionResult = engine.run(code)
  assert result.success, f"Failed: {result.errors}"
  assert "sw.constant" in result.code
  assert "value = 1" in result.code


def test_mlir_to_python(base_engine: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verify `base_engine` emits `1` for MLIR to python.

  Args:
      base_engine: Input `base_engine` parameter value (`typing.Callable[[str, str], ASTEngine]`).
  """
  engine = base_engine("mlir", "jax")
  mlir_code: str = '%0 = "sw.constant"() {value = 1}'
  result: ConversionResult = engine.run(mlir_code)
  assert result.success, f"Failed: {result.errors}"
  assert "1" in result.code


def test_mlir_to_mlir_roundtrip(base_engine: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verify `base_engine` emits `sw.op` for MLIR to MLIR roundtrip.

  Args:
      base_engine: Input `base_engine` parameter value (`typing.Callable[[str, str], ASTEngine]`).
  """
  engine = base_engine("mlir", "mlir")
  mlir_code: str = '%0 = "sw.op"() {type = "util.noop"}'
  result: ConversionResult = engine.run(mlir_code)
  assert result.success, f"Failed: {result.errors}"
  assert "sw.op" in result.code
  assert "util.noop" in result.code
