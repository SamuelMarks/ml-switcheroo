"""Test suite for the Engine Switching module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.frameworks import register_framework

register_framework("mlir")(MlirAdapter)


@pytest.fixture
def base_engine():
  """Provides a mock base engine for testing."""
  semantics = MagicMock(spec=SemanticsManager)
  semantics.get_framework_config.return_value = {}
  semantics.get_import_map.return_value = {}
  semantics.get_framework_aliases.return_value = {}

  def get_def_side_effect(name):
    """Gets def side effect."""
    return None

  semantics.get_definition.side_effect = get_def_side_effect

  def create(source, target):
    """Creates ."""
    config = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=False, enable_import_fixer=False)
    config.validation_report = None
    return ASTEngine(semantics, config)

  return create


def test_python_to_mlir(base_engine):
  """Verifies the behavior of python to MLIR."""
  engine = base_engine("torch", "mlir")
  code = "x = 1"
  result = engine.run(code)
  assert result.success, f"Failed: {result.errors}"
  assert "sw.constant" in result.code
  assert "value = 1" in result.code


def test_mlir_to_python(base_engine):
  """Verifies the behavior of MLIR to python."""
  engine = base_engine("mlir", "jax")
  mlir_code = '%0 = "sw.constant"() {value = 1}'
  result = engine.run(mlir_code)
  assert result.success, f"Failed: {result.errors}"
  assert "1" in result.code


def test_mlir_to_mlir_roundtrip(base_engine):
  """Verifies the behavior of MLIR to MLIR roundtrip."""
  engine = base_engine("mlir", "mlir")
  mlir_code = '%0 = "sw.op"() {type = "util.noop"}'
  result = engine.run(mlir_code)
  assert result.success, f"Failed: {result.errors}"
  assert "sw.op" in result.code
  assert "util.noop" in result.code
