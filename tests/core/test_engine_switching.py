"""
Tests for ASTEngine Source/Target Switching Logic.

Verifies:
1. Python -> MLIR (Target Switching).
2. MLIR -> Python (Source Switching).
3. MLIR -> MLIR (Roundtrip).
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.frameworks import register_framework

# Ensure MLIR adapter registered for tests
register_framework("mlir")(MlirAdapter)


@pytest.fixture
def base_engine():
  semantics = MagicMock(spec=SemanticsManager)
  semantics.get_framework_config.return_value = {}
  semantics.get_import_map.return_value = {}
  semantics.get_framework_aliases.return_value = {}

  # IMPORTANT: Mock get_definition to return None for unmapped ops like 'util.noop'
  # to prevent PivotRewriter from iterating over a MagicMock iterator, which causes crash.
  def get_def_side_effect(name):
    return None

  semantics.get_definition.side_effect = get_def_side_effect

  def create(source, target):
    # Must disable ImportFixer for MLIR source/target to avoid scanner crashes on non-python trees
    config = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=False, enable_import_fixer=False)
    # Manually ensure no validation report is loaded to prevent mock issues
    config.validation_report = None
    return ASTEngine(semantics, config)

  return create


def test_python_to_mlir(base_engine):
  """
  Scenario: User converts Python code to MLIR.
  Input: x = 1
  Output: MLIR Text containing sw.constant
  """
  engine = base_engine("torch", "mlir")
  code = "x = 1"

  result = engine.run(code)

  assert result.success, f"Failed: {result.errors}"
  # The output should be MLIR text from the MlirAdapter's emitter wrapper
  assert "sw.constant" in result.code
  assert "value = 1" in result.code


def test_mlir_to_python(base_engine):
  """
  Scenario: User converts MLIR code to Python.
  """
  engine = base_engine("mlir", "jax")
  # Use canonical spacing to ensure tokenizer handles it cleanly
  mlir_code = '%0 = "sw.constant"() {value = 1}'

  result = engine.run(mlir_code)

  assert result.success, f"Failed: {result.errors}"
  # generated code for unused constant '1' is '1' statement
  assert "1" in result.code


def test_mlir_to_mlir_roundtrip(base_engine):
  """
  Scenario: MLIR -> Python AST -> MLIR
  """
  engine = base_engine("mlir", "mlir")
  mlir_code = '%0 = "sw.op"() {type = "util.noop"}'

  result = engine.run(mlir_code)

  assert result.success, f"Failed: {result.errors}"
  # The bridge reconstructs it.
  assert "sw.op" in result.code
  assert "util.noop" in result.code
