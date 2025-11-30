"""
Tests for Compatibility Matrix Generation.

Verifies:
1. `render()` outputs formatted text to stdout.
2. `get_json()` returns structured data suitable for API responses.
3. Status icons are correctly computed based on variant presence and plugin flags.
"""

from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """
  Mock Manager that bypasses file loading to provide deterministic test data.
  """

  def __init__(self) -> None:
    """Initialize with predefined operation data."""
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
  assert "TORCH" in stdout

  # 2. Check Row Content
  assert "abs" in stdout
  assert "complex_op" in stdout


def test_matrix_get_json_structure() -> None:
  """
  Verify `get_json` returns a list of dicts with correct keys and values.
  """
  semantics = MockSemantics()
  matrix = CompatibilityMatrix(semantics)

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
  # Ensure unsupported engines are X
  assert missing_row["tensorflow"] == "❌"


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
