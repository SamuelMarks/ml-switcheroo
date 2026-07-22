"""Test suite for the Matrix module."""

import pytest
from unittest.mock import MagicMock, patch
from rich.console import Console
from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


class MockMatrixSemantics(SemanticsManager):
  """Mock Matrix Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockMatrixSemantics instance."""
    self.data = {
      "Abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "ComplexOp": {"variants": {"torch": {"api": "torch.complex"}, "jax": {"requires_plugin": "magic_fix"}}},
      "MissingOp": {"variants": {"torch": {"api": "torch.foo"}}},
    }
    self._key_origins = {"Abs": "array", "ComplexOp": "neural", "MissingOp": "extras"}

  def get_known_apis(self) -> dict:
    """Mock implementation of get known apis."""
    return self.data


@pytest.fixture
def semantics() -> SemanticsManager:
  """Provides a mock semantics for testing."""
  return MockMatrixSemantics()


@pytest.fixture
def matrix(semantics) -> CompatibilityMatrix:
  """Provides a mock matrix for testing."""
  mat = CompatibilityMatrix(semantics)
  mat.console = Console(file=None, force_terminal=True, width=200, record=True)
  return mat


def test_status_icon_resolution(matrix):
  """Verifies the behavior of status icon resolution."""
  assert matrix._get_status_icon({"api": "foo"}) == "✅"
  assert matrix._get_status_icon({"requires_plugin": "foo"}) == "🧩"
  assert matrix._get_status_icon(None) == "❌"
  assert matrix._get_status_icon({}) == "❌"


@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_dynamic_column_sorting(mock_get_adapter, mock_avail, matrix):
  """Verifies the behavior of dynamic column sorting."""
  mock_avail.return_value = ["beta", "alpha", "gamma"]
  adapter_alpha = MagicMock(ui_priority=10, inherits_from=None)
  adapter_beta = MagicMock(ui_priority=50, inherits_from=None)
  adapter_gamma = MagicMock(ui_priority=5, inherits_from=None)

  def get_adp(name):
    """Gets adp."""
    if name == "alpha":
      return adapter_alpha
    if name == "beta":
      return adapter_beta
    if name == "gamma":
      return adapter_gamma
    return None

  mock_get_adapter.side_effect = get_adp
  rows = matrix.get_json()
  engines = matrix._get_sorted_engines()
  assert engines == ["gamma", "alpha", "beta"]
  row0 = rows[0]
  assert "gamma" in row0
  assert "alpha" in row0
  assert "beta" in row0


@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_render_output_contains_headers(mock_get_adapter, mock_avail, matrix):
  """Renders output contains headers."""
  mock_avail.return_value = ["torch", "jax"]
  mock_get_adapter.side_effect = lambda n: MagicMock(ui_priority=0 if n == "torch" else 10, inherits_from=None)
  matrix.render()
  output = matrix.console.export_text()
  assert "TORCH" in output
  assert "JAX" in output
  assert "Abs" in output
  assert "Array" in output
  assert "✅" in output
  assert "🧩" in output
  assert "❌" in output


@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_inheritance_hiding_logic(mock_get_adapter, mock_avail, matrix):
  """Verifies the behavior of inheritance hiding logic."""
  mock_avail.return_value = ["jax", "flax_nnx"]
  adp_jax = MagicMock(ui_priority=10, inherits_from=None)
  adp_flax = MagicMock(ui_priority=15, inherits_from="jax")
  mock_get_adapter.side_effect = lambda n: adp_jax if n == "jax" else adp_flax
  engines = matrix._get_sorted_engines()
  assert engines[-1] == "flax_nnx"
  assert "jax" in engines
