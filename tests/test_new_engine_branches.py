"""Tests for `ASTEngine` initialization and execution with intermediate IR and configs."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine


def test_astengine_init_and_execution_branches() -> None:
  """Verify `ASTEngine` initializes and executes conversions across config and intermediate variants."""
  engine1: ASTEngine = ASTEngine(source="torch", target="jax", intermediate="onnx")
  assert engine1.config.intermediate == "onnx"
  res1 = engine1.run("import torch\nx = torch.abs(y)\n")
  assert res1.success is True
  assert "jnp.abs(y)" in res1.code

  cfg1: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  engine2: ASTEngine = ASTEngine(config=cfg1)
  assert engine2.config.intermediate == "mlir"
  res2 = engine2.run("import torch\nx = torch.add(a, b)\n")
  assert res2.success is True
  assert "jax.lax.add(a, b)" in res2.code or "jnp.add(a, b)" in res2.code

  cfg2: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine3 = ASTEngine(config=cfg2)
  assert not cfg2.validation_report
  res3 = engine3.run("import torch\nx = torch.sin(z)\n")
  assert res3.success is True
  assert "jnp.sin(z)" in res3.code
