"""Test suite for the Matrix module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


class MockMatrixSemantics(SemanticsManager):
  """Test helper class MockMatrixSemantics."""

  def __init__(self) -> None:
    """Initializes the MockMatrixSemantics instance."""
    self.data = {
      "Abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "ComplexOp": {"variants": {"torch": {"api": "torch.complex"}, "jax": {"requires_plugin": "magic_fix"}}},
      "MissingOp": {"variants": {"torch": {"api": "torch.foo"}}},
    }
    self._key_origins: dict[str, str] = {"Abs": "array", "ComplexOp": "neural", "MissingOp": "extras"}

  def get_known_apis(self) -> dict[str, JsonValue]:  # type: ignore[override]
    """Mock implementation of get known apis.

    Returns:
        Computed test result or fixture value.
    """
    return self.data  # type: ignore[return-value]


@pytest.fixture
def semantics() -> MockMatrixSemantics:
  """Execute semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return MockMatrixSemantics()


@pytest.fixture
def matrix(semantics: MockMatrixSemantics) -> CompatibilityMatrix:
  """Execute matrix helper logic.

  Args:
      semantics: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  mat = CompatibilityMatrix(semantics)
  mat.console = Console(file=None, force_terminal=True, width=200, record=True)
  return mat


def test_status_icon_resolution(matrix: CompatibilityMatrix) -> None:
  """Verifies the behavior of status icon resolution.

  Args:
      matrix: Test argument or fixture.
  """
  assert matrix._get_status_icon({"api": "foo"}) == "✅"
  assert matrix._get_status_icon({"requires_plugin": "foo"}) == "🧩"
  assert matrix._get_status_icon(None) == "❌"
  assert matrix._get_status_icon({}) == "❌"


@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_dynamic_column_sorting(mock_get_adapter: MagicMock, mock_avail: MagicMock, matrix: CompatibilityMatrix) -> None:
  """Verifies the behavior of dynamic column sorting.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_avail: Test argument or fixture.
      matrix: Test argument or fixture.
  """
  mock_avail.return_value = ["beta", "alpha", "gamma"]
  adapter_alpha = MagicMock(ui_priority=10, inherits_from=None)
  adapter_beta = MagicMock(ui_priority=50, inherits_from=None)
  adapter_gamma = MagicMock(ui_priority=5, inherits_from=None)

  def get_adp(name: str) -> typing.Optional[MagicMock]:
    """Gets adp.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "alpha":
      return adapter_alpha
    if name == "beta":
      return adapter_beta
    if name == "gamma":
      return adapter_gamma
    return None

  mock_get_adapter.side_effect = get_adp
  rows: list[dict[str, str]] = matrix.get_json()
  engines: list[str] = matrix._get_sorted_engines()
  assert engines == ["gamma", "alpha", "beta"]
  row0: dict[str, str] = rows[0]
  assert "gamma" in row0
  assert "alpha" in row0
  assert "beta" in row0


@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_render_output_contains_headers(
  mock_get_adapter: MagicMock, mock_avail: MagicMock, matrix: CompatibilityMatrix
) -> None:
  """Renders output contains headers.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_avail: Test argument or fixture.
      matrix: Test argument or fixture.
  """
  mock_avail.return_value = ["torch", "jax"]
  mock_get_adapter.side_effect = lambda n: MagicMock(ui_priority=0 if n == "torch" else 10, inherits_from=None)
  matrix.render()
  output: str = matrix.console.export_text()
  assert "TORCH" in output
  assert "JAX" in output
  assert "Abs" in output
  assert "Array" in output
  assert "✅" in output
  assert "🧩" in output
  assert "❌" in output


@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_inheritance_hiding_logic(
  mock_get_adapter: MagicMock, mock_avail: MagicMock, matrix: CompatibilityMatrix
) -> None:
  """Verifies the behavior of inheritance hiding logic.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_avail: Test argument or fixture.
      matrix: Test argument or fixture.
  """
  mock_avail.return_value = ["jax", "flax_nnx"]
  adp_jax = MagicMock(ui_priority=10, inherits_from=None)
  adp_flax = MagicMock(ui_priority=15, inherits_from="jax")
  mock_get_adapter.side_effect = lambda n: adp_jax if n == "jax" else adp_flax
  engines: list[str] = matrix._get_sorted_engines()
  assert engines[-1] == "flax_nnx"
  assert "jax" in engines
