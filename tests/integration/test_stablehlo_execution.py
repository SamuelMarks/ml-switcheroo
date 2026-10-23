"""Tests for StableHLO execution and parity."""

import types

import numpy as np
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager

jax = pytest.importorskip("jax")
xla_bridge: types.ModuleType | None = None
try:
  from jax.lib import xla_bridge as _xla_bridge

  xla_bridge = _xla_bridge
except ImportError:
  pass


def requires_pjrt() -> bool:
  """Check if JAX PJRT CPU backend is available.

  Returns:
      Resulting `bool` from requires pjrt.
  """
  try:
    assert xla_bridge is not None
    getattr(xla_bridge, "get_backend")()
    return True
  except Exception:
    return False


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Execute semantics helper logic.

  Returns:
      Configured `semantics` fixture value (SemanticsManager).
  """
  mgr = SemanticsManager()
  return mgr


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_abs_execution(semantics: SemanticsManager) -> None:
  """Verify that PJRT execution wrapping works properly for stablehlo.abs.

  Args:
      semantics: SemanticsManager instance providing operation and trait definitions.
  """
  # This verifies PJRT execution wrapping works properly
  mlir_code = """
module {
  func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
    %0 = "stablehlo.abs"(%x) : (tensor<3xf32>) -> tensor<3xf32>
    func.return %0 : tensor<3xf32>
  }
}
"""
  assert xla_bridge is not None
  _client = getattr(xla_bridge, "get_backend")()
  _executable = _client.compile(mlir_code)

  def forward(
    x: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
  ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Execute the compiled StableHLO module for abs.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    buf = _client.buffer_from_pyval(x)
    res = _executable.execute([buf])
    return np.asarray(res[0], dtype=np.float32)

  x_in = np.array([-1.5, 2.0, -3.14], dtype=np.float32)
  y_out = forward(x_in)
  np.testing.assert_allclose(y_out, np.abs(x_in))


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_math_parity(semantics: SemanticsManager) -> None:
  """Verify stablehlo math parity behavior.

  Args:
      semantics: SemanticsManager instance providing operation and trait definitions.
  """
  mlir_code = """
module {
  func.func @main(%x: tensor<3xf32>, %y: tensor<3xf32>) -> tensor<3xf32> {
    %0 = "stablehlo.multiply"(%y, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %1 = "stablehlo.add"(%x, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %1 : tensor<3xf32>
  }
}
"""
  assert xla_bridge is not None
  _client = getattr(xla_bridge, "get_backend")()
  _executable = _client.compile(mlir_code)

  def forward(
    x: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    y: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
  ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Execute the compiled StableHLO module for math operations.

    Args:
        x: Primary input tensor or array operand.
        y: Secondary input tensor or target operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    buf_x = _client.buffer_from_pyval(x)
    buf_y = _client.buffer_from_pyval(y)
    res = _executable.execute([buf_x, buf_y])
    return np.asarray(res[0], dtype=np.float32)

  x_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  y_in = np.array([4.0, 5.0, 6.0], dtype=np.float32)
  y_out = forward(x_in, y_in)
  np.testing.assert_allclose(y_out, x_in + (y_in * x_in))


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_while_parity(semantics: SemanticsManager) -> None:
  """Verify stablehlo while parity behavior.

  Args:
      semantics: SemanticsManager instance providing operation and trait definitions.
  """
  # This is a basic structural test for while compilation parity
  code = "import torch\ndef forward(x: torch.Tensor, count: torch.Tensor):\n    while count:\n        x = torch.abs(x)\n    return x\n"
  config = RuntimeConfig(source_framework="torch", target_framework="stablehlo", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)
  assert result.success
  assert "stablehlo.while" in result.code
