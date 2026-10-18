"""Tests for StableHLO execution and parity."""

import typing

import numpy as np
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager

jax: typing.Any = pytest.importorskip("jax")
xla_bridge: typing.Any = None
try:
  from jax.lib import xla_bridge as _xla_bridge

  xla_bridge = _xla_bridge
except ImportError:
  pass


def requires_pjrt() -> typing.Any:
  """Check if JAX PJRT CPU backend is available.

  Returns:
      Computed test result or fixture value.
  """
  try:
    xla_bridge.get_backend()
    return True
  except Exception:
    return False


@pytest.fixture(scope="module")
def semantics() -> typing.Any:
  """Execute semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr: typing.Any = SemanticsManager()
  return mgr


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_abs_execution(semantics: typing.Any) -> None:
  """Verify that PJRT execution wrapping works properly for stablehlo.abs.

  Args:
      semantics: Test argument or fixture.
  """
  # This verifies PJRT execution wrapping works properly
  mlir_code: typing.Any = """
module {
  func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
    %0 = "stablehlo.abs"(%x) : (tensor<3xf32>) -> tensor<3xf32>
    func.return %0 : tensor<3xf32>
  }
}
"""
  _client: typing.Any = xla_bridge.get_backend()
  _executable: typing.Any = _client.compile(mlir_code)

  def forward(x: typing.Any) -> typing.Any:
    """Execute the compiled StableHLO module for abs.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    buf: typing.Any = _client.buffer_from_pyval(x)
    res: typing.Any = _executable.execute([buf])
    return np.asarray(res[0])

  x_in: typing.Any = np.array([-1.5, 2.0, -3.14], dtype=np.float32)
  y_out: typing.Any = forward(x_in)
  np.testing.assert_allclose(y_out, np.abs(x_in))


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_math_parity(semantics: typing.Any) -> None:
  """Verify stablehlo math parity behavior.

  Args:
      semantics: Test argument or fixture.
  """
  mlir_code: typing.Any = """
module {
  func.func @main(%x: tensor<3xf32>, %y: tensor<3xf32>) -> tensor<3xf32> {
    %0 = "stablehlo.multiply"(%y, %x) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %1 = "stablehlo.add"(%x, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    func.return %1 : tensor<3xf32>
  }
}
"""
  _client: typing.Any = xla_bridge.get_backend()
  _executable: typing.Any = _client.compile(mlir_code)

  def forward(x: typing.Any, y: typing.Any) -> typing.Any:
    """Execute the compiled StableHLO module for math operations.

    Args:
        x: Test argument or fixture.
        y: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    buf_x: typing.Any = _client.buffer_from_pyval(x)
    buf_y: typing.Any = _client.buffer_from_pyval(y)
    res: typing.Any = _executable.execute([buf_x, buf_y])
    return np.asarray(res[0])

  x_in: typing.Any = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  y_in: typing.Any = np.array([4.0, 5.0, 6.0], dtype=np.float32)
  y_out: typing.Any = forward(x_in, y_in)
  np.testing.assert_allclose(y_out, x_in + (y_in * x_in))


@pytest.mark.skipif(not requires_pjrt(), reason="JAX PJRT cpu backend not available.")
def test_stablehlo_while_parity(semantics: typing.Any) -> None:
  """Verify stablehlo while parity behavior.

  Args:
      semantics: Test argument or fixture.
  """
  # This is a basic structural test for while compilation parity
  code: typing.Any = "import torch\ndef forward(x: torch.Tensor, count: torch.Tensor):\n    while count:\n        x = torch.abs(x)\n    return x\n"
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="stablehlo", strict_mode=True)
  engine: typing.Any = ASTEngine(semantics=semantics, config=config)
  result: typing.Any = engine.run(code)
  assert result.success
  assert "stablehlo.while" in result.code
