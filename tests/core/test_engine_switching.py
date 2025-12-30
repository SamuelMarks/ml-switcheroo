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


@pytest.fixture
def base_engine():
  semantics = MagicMock(spec=SemanticsManager)
  # Ensure safe defaults for lookups
  semantics.get_framework_config.return_value = {}
  semantics.get_import_map.return_value = {}
  semantics.get_framework_aliases.return_value = {}

  def create(source, target):
    config = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=False)
    return ASTEngine(semantics, config)

  return create


def test_python_to_mlir(base_engine):
  """
  Scenario: User converts Python code to MLIR.
  Input: x = 1
  Output: MLIR Text with sw.constant
  """
  engine = base_engine("torch", "mlir")
  code = "x = 1"

  result = engine.run(code)

  assert result.success
  assert "sw.constant" in result.code
  assert "value = 1" in result.code
  assert result.has_errors is False


def test_mlir_to_python(base_engine):
  """
  Scenario: User converts MLIR code to Python (Default jax target).
  Input: %0 = sw.constant {value=1}
  Output: 1 (Expression Statement via Void Suppression)
  """
  # Note: rewriter needs config, so target='jax' means standard rewriter runs.
  # The generated python code is generic ("_0 = 1"), which might not trigger rewriter changes.
  engine = base_engine("mlir", "jax")

  mlir_code = '%0 = "sw.constant"() {value = 1}'

  result = engine.run(mlir_code)

  assert result.success
  # Generator creates just "1\n" as it is unused and suppressing assignments
  assert "1" in result.code
  assert "=" not in result.code


def test_mlir_to_mlir_roundtrip(base_engine):
  """
  Scenario: MLIR -> Python AST -> MLIR
  Input: %0 = "sw.op"() : i32
  Output: Should generally match input structure (normalized).
  """
  engine = base_engine("mlir", "mlir")

  mlir_code = '%0 = "sw.op"() {type = "util.noop"} : ()'

  result = engine.run(mlir_code)

  assert result.success
  assert "sw.op" in result.code
  assert "util.noop" in result.code
