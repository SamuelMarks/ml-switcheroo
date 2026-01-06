"""
Tests for Compatibility Matrix UI (Dynamic Generation).

Verifies:
1. Dynamic Column Discovery using `get_framework_priority_order`.
2. Sorting Priorities (Mocked Adapters).
3. Status Icon Logic (Direct vs Plugin vs Missing).
4. JSON Export Structure.
"""

import pytest
from unittest.mock import MagicMock, patch
from rich.console import Console

from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


class MockMatrixSemantics(SemanticsManager):
  """
  Mock Manager providing deterministic API definitions for the matrix.
  Bypasses file system loading.
  """

  def __init__(self) -> None:
    # Skip super init
    self.data = {
      "Abs": {
        "variants": {
          "torch": {"api": "torch.abs"},
          "jax": {"api": "jax.numpy.abs"},
        }
      },
      "ComplexOp": {
        "variants": {
          "torch": {"api": "torch.complex"},
          "jax": {"requires_plugin": "magic_fix"},  # Plugin
        }
      },
      "MissingOp": {
        "variants": {
          "torch": {"api": "torch.foo"}
          # JAX Missing
        }
      },
    }
    self._key_origins = {"Abs": "array", "ComplexOp": "neural", "MissingOp": "extras"}

  def get_known_apis(self) -> dict:
    return self.data


@pytest.fixture
def semantics() -> SemanticsManager:
  return MockMatrixSemantics()


@pytest.fixture
def matrix(semantics) -> CompatibilityMatrix:
  mat = CompatibilityMatrix(semantics)
  # Use capture console
  mat.console = Console(file=None, force_terminal=True, width=200, record=True)
  return mat


def test_status_icon_resolution(matrix):
  """Verify icon logic maps states correctly."""
  # Direct
  assert matrix._get_status_icon({"api": "foo"}) == "✅"
  # Plugin
  assert matrix._get_status_icon({"requires_plugin": "foo"}) == "🧩"
  # Missing / None
  assert matrix._get_status_icon(None) == "❌"
  assert matrix._get_status_icon({}) == "❌"


# Fix: Patch where it is defined since it is dynamically imported in config.py
@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_dynamic_column_sorting(mock_get_adapter, mock_avail, matrix):
  """
  Verify that columns are ordered by Adapter UI priority.
  """
  # 1. Setup Mock Registry
  # Unsorted list
  mock_avail.return_value = ["beta", "alpha", "gamma"]

  # 2. Setup Adapters with priorities
  # alpha=10, beta=50, gamma=5
  # Expected Order: Gamma (5), Alpha (10), Beta (50)
  adapter_alpha = MagicMock(ui_priority=10, inherits_from=None)
  adapter_beta = MagicMock(ui_priority=50, inherits_from=None)
  adapter_gamma = MagicMock(ui_priority=5, inherits_from=None)

  def get_adp(name):
    if name == "alpha":
      return adapter_alpha
    if name == "beta":
      return adapter_beta
    if name == "gamma":
      return adapter_gamma
    return None

  mock_get_adapter.side_effect = get_adp

  # 3. Get JSON
  # get_json() calls _get_sorted_engines() which calls config.get_framework_priority_order()
  rows = matrix.get_json()

  # 4. Infer order from keys presence in dict
  # But rows are dicts. We need to check if the sorting function works.
  engines = matrix._get_sorted_engines()
  assert engines == ["gamma", "alpha", "beta"]

  # Confirm JSON has these keys
  row0 = rows[0]
  assert "gamma" in row0
  assert "alpha" in row0
  assert "beta" in row0


# Fix: Patch where defined
@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_render_output_contains_headers(mock_get_adapter, mock_avail, matrix):
  """
  Verify Rich Table rendering includes dynamic headers.
  """
  mock_avail.return_value = ["torch", "jax"]
  # Simple priority to enforce order
  mock_get_adapter.side_effect = lambda n: MagicMock(ui_priority=0 if n == "torch" else 10, inherits_from=None)

  matrix.render()
  output = matrix.console.export_text()

  # Check Headers (Uppercase)
  assert "TORCH" in output
  assert "JAX" in output

  # Check Content
  assert "Abs" in output
  assert "Array" in output
  assert "✅" in output  # Torch/Jax match
  assert "🧩" in output  # ComplexOp Jax
  assert "❌" in output  # MissingOp Jax


# Fix: Patch where defined
@patch("ml_switcheroo.frameworks.base.available_frameworks")
@patch("ml_switcheroo.frameworks.base.get_adapter")
def test_inheritance_hiding_logic(mock_get_adapter, mock_avail, matrix):
  """
  Verify that frameworks inheriting from others (Children/Flavours) are
  sorted to the end (priority 9999) to keep the main table clean.
  """
  mock_avail.return_value = ["jax", "flax_nnx"]

  # Jax = 10, Flax = inherits
  adp_jax = MagicMock(ui_priority=10, inherits_from=None)
  adp_flax = MagicMock(ui_priority=15, inherits_from="jax")

  mock_get_adapter.side_effect = lambda n: adp_jax if n == "jax" else adp_flax

  engines = matrix._get_sorted_engines()

  # Even though Flax has priority 15, inheritance logic in config pushes it to end
  assert engines[-1] == "flax_nnx"
  assert "jax" in engines
