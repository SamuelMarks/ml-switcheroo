"""Tests for new engine branches."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig


def test_astengine_init_missing_branches():
  """Test ASTEngine init missing branches."""
  engine1 = ASTEngine(source="torch", target="jax", intermediate="onnx")
  assert engine1.config.intermediate == "onnx"

  cfg1 = RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  engine2 = ASTEngine(config=cfg1)
  assert engine2.config.intermediate == "mlir"

  cfg2 = RuntimeConfig(source_framework="torch", target_framework="jax")
  ASTEngine(config=cfg2)
  assert not cfg2.validation_report
