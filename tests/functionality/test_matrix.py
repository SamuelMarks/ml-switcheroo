"""
Tests for Compatibility Matrix Generation.

Verifies:
1. `render()` outputs formatted text to stdout.
2. `get_json()` returns structured data.
3. Logic sorts based on adapter dynamic priorities.
"""

from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """
  Mock Manager that bypasses file loading.
  """

  def __init__(self) -> None:
    # Skip file loading
    self.data = {
      "abs": {
        "variants": {
          "torch": {"api": "torch.abs"},
          "jax": {"api": "jax.numpy.abs"},
        }
      },
      "complex_op": {
        "variants": {
          "torch": {"api": "torch.complex"},
          "jax": {"requires_plugin": "magic_fix"},
        }
      },
      "missing_op": {"variants": {"torch": {"api": "torch.foo"}}},
    }
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}

  def get_known_apis(self) -> dict:
    """Return the mocked data dictionary."""
    return self.data


def test_matrix_rendering(capsys) -> None:
  """
  Verify that the matrix renders to stdout and contains expected text.
  """
  semantics = MockSemantics()
  matrix = CompatibilityMatrix(semantics)

  # Render
  matrix.render()

  # Capture Output
  captured = capsys.readouterr()
  stdout = captured.out

  # Assertions
  # 1. Check Table Headers
  assert "Operation" in stdout
  assert "Tier" in stdout
  # Verify at least one framework column is present (assuming frameworks exist in registry)

  # 2. Check Row Content
  assert "abs" in stdout
  assert "complex_op" in stdout


def test_matrix_get_json_structure() -> None:
  """
  Verify `get_json` returns a list of dicts with correct keys and values.
  """
  semantics = MockSemantics()
  matrix = CompatibilityMatrix(semantics)

  # We mock available_frameworks to ensure deterministic columns for this test
  # The updated logic calls get_framework_priority_order which calls available_frameworks
  with patch("ml_switcheroo.config.available_frameworks", return_value=["torch", "jax"]):
    # We also need to mock get_adapter to return adapter objects with ui_priority property
    mock_torch = MagicMock()
    mock_torch.ui_priority = 0
    mock_jax = MagicMock()
    mock_jax.ui_priority = 10

    def mock_get(name):
      if name == "torch":
        return mock_torch
      if name == "jax":
        return mock_jax
      return None

    # Patch the source definition used by config.py
    with patch("ml_switcheroo.frameworks.get_adapter", side_effect=mock_get):
      result = matrix.get_json()

  assert isinstance(result, list)
  assert len(result) == 3

  # Find specific rows
  abs_row = next(r for r in result if r["operation"] == "abs")
  complex_row = next(r for r in result if r["operation"] == "complex_op")
  missing_row = next(r for r in result if r["operation"] == "missing_op")

  # 1. Standard Mapping
  assert abs_row["torch"] == "✅"
  assert abs_row["jax"] == "✅"

  # 2. Plugin Mapping
  assert complex_row["torch"] == "✅"
  assert complex_row["jax"] == "🧩"  # Plugin icon

  # 3. Missing Variant
  assert missing_row["torch"] == "✅"
  assert missing_row["jax"] == "❌"


def test_status_icon_logic() -> None:
  """
  Verify internal status resolution logic directly.
  """
  semantics = MockSemantics()
  matrix = CompatibilityMatrix(semantics)

  # Standard
  assert matrix._get_status_icon({"api": "foo"}) == "✅"

  # Plugin
  assert matrix._get_status_icon({"requires_plugin": "foo"}) == "🧩"

  # Missing / None
  assert matrix._get_status_icon(None) == "❌"
  assert matrix._get_status_icon({}) == "❌"


def test_sorting_logic_priorities():
  """
  Verify that _get_sorted_engines sorts based on dynamic adapter priority.
  """
  semantics = MockSemantics()
  matrix = CompatibilityMatrix(semantics)

  # Inputs
  fws = ["alpha", "torch", "beta", "jax"]

  # Create mock adapters with priorities
  # Torch (0), Jax (10), Alpha (50), Beta (Unknown/999)
  mock_torch = MagicMock()
  mock_torch.ui_priority = 0

  mock_jax = MagicMock()
  mock_jax.ui_priority = 10

  mock_alpha = MagicMock()
  mock_alpha.ui_priority = 50

  # Beta has no adapter logic (simulate unknown/legacy)

  def mock_get(name):
    if name == "torch":
      return mock_torch
    if name == "jax":
      return mock_jax
    if name == "alpha":
      return mock_alpha
    return None  # Beta

  with patch("ml_switcheroo.config.available_frameworks", return_value=fws):
    # Patch the source definition used by config.py
    with patch("ml_switcheroo.frameworks.get_adapter", side_effect=mock_get):
      sorted_list = matrix._get_sorted_engines()

  # Expected: torch (0), jax (10), alpha (50), beta (999)
  assert sorted_list[0] == "torch"
  assert sorted_list[1] == "jax"
  assert sorted_list[2] == "alpha"
  assert sorted_list[3] == "beta"
