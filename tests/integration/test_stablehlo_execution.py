"""Tests for StableHLO execution and parity."""

import pytest
import numpy as np
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

jax = pytest.importorskip("jax")
try:
  from jax.lib import xla_bridge
except ImportError:
  xla_bridge = None


def requires_pjrt():
  """Check if JAX PJRT CPU backend is available."""
  try:
    xla_bridge.get_backend()
    return True
  except Exception:
    return False


@pytest.fixture(scope="module")
def semantics():
  """Provide a SemanticsManager instance for the tests."""
  mgr = SemanticsManager()
  return mgr


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_abs_execution(semantics):
  """Verify that PJRT execution wrapping works properly for stablehlo.abs."""
  # This verifies PJRT execution wrapping works properly
  mlir_code = """
module {
  func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
    %0 = "stablehlo.abs"(%x) : (tensor<3xf32>) -> tensor<3xf32>
    func.return %0 : tensor<3xf32>
  }
}
"""
  _client = xla_bridge.get_backend()
  _executable = _client.compile(mlir_code)

  def forward(x):
    """Execute the compiled StableHLO module for abs."""
    buf = _client.buffer_from_pyval(x)
    res = _executable.execute([buf])
    return np.asarray(res[0])

  x_in = np.array([-1.5, 2.0, -3.14], dtype=np.float32)
  y_out = forward(x_in)
  np.testing.assert_allclose(y_out, np.abs(x_in))


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_math_parity(semantics):
  """Test StableHLO execution parity for basic math operations."""
  mlir_code = """
module {
  func.func @main(%x: tensor<3xf32>, %y: tensor<3xf32>) -> tensor<3xf32> {
    %0 = "stablehlo.multiply"(%y, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %1 = "stablehlo.add"(%x, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %1 : tensor<3xf32>
  }
}
"""
  _client = xla_bridge.get_backend()
  _executable = _client.compile(mlir_code)

  def forward(x, y):
    """Execute the compiled StableHLO module for math operations."""
    buf_x = _client.buffer_from_pyval(x)
    buf_y = _client.buffer_from_pyval(y)
    res = _executable.execute([buf_x, buf_y])
    return np.asarray(res[0])

  x_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  y_in = np.array([4.0, 5.0, 6.0], dtype=np.float32)
  y_out = forward(x_in, y_in)
  np.testing.assert_allclose(y_out, x_in + (y_in * x_in))


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_while_parity(semantics):
  """Test the structural generation of stablehlo.while loop."""
  # This is a basic structural test for while compilation parity
  code = "import torch\ndef forward(x: torch.Tensor, count: torch.Tensor):\n    while count:\n        x = torch.abs(x)\n    return x\n"
  config = RuntimeConfig(source_framework="torch", target_framework="stablehlo", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)
  assert result.success
  assert "stablehlo.while" in result.code
