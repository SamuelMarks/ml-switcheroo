"""Test module."""

from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


def test_compatibility_matrix_get_json() -> None:
  """Test element."""
  semantics = SemanticsManager()

  # Mock data
  semantics.get_known_apis = MagicMock(
    return_value={
      "Conv2d": {"variants": {"torch": {"api": "torch.nn.Conv2d"}, "jax": {"requires_plugin": "flax_conv"}}},
      "add": {"variants": {"torch": {"api": "torch.add"}}},
    }
  )

  semantics._key_origins = {"Conv2d": "neural_net", "add": "math"}

  matrix = CompatibilityMatrix(semantics)

  with patch.object(matrix, "_get_sorted_engines", return_value=["torch", "jax"]):
    data: list[dict[str, str]] = matrix.get_json()

    assert len(data) == 2

    # Test Conv2d row
    conv_row: dict[str, str] = next(r for r in data if r["operation"] == "Conv2d")
    assert conv_row["tier"] == "Neural Net"
    assert conv_row["torch"] == "✅"
    assert conv_row["jax"] == "🧩"

    # Test add row
    add_row: dict[str, str] = next(r for r in data if r["operation"] == "add")
    assert add_row["tier"] == "Math"
    assert add_row["torch"] == "✅"
    assert add_row["jax"] == "❌"


def test_compatibility_matrix_render() -> None:
  """Test element."""
  semantics = SemanticsManager()
  matrix = CompatibilityMatrix(semantics)

  mock_json: list[dict[str, str]] = [{"operation": "test_op", "tier": "Test", "torch": "✅", "jax": "❌"}]

  with patch.object(matrix, "get_json", return_value=mock_json):
    with patch.object(matrix, "_get_sorted_engines", return_value=["torch", "jax"]):
      with patch.object(matrix.console, "print") as mock_print:
        matrix.render()

        mock_print.assert_called_once()
        table = mock_print.call_args[0][0]
        assert table.title == "ml-switcheroo Compatibility Matrix"
        # Check columns
        cols: list[str] = [c.header for c in table.columns]
        assert cols == ["Operation", "Tier", "TORCH", "JAX"]


def test_compatibility_matrix_get_status_icon() -> None:
  """Test element."""
  matrix = CompatibilityMatrix(SemanticsManager())

  assert matrix._get_status_icon(None) == "❌"
  assert matrix._get_status_icon({}) == "❌"
  assert matrix._get_status_icon({"api": "some_api"}) == "✅"
  assert matrix._get_status_icon({"requires_plugin": "plugin"}) == "🧩"


def test_compatibility_matrix_get_sorted_engines() -> None:
  """Test element."""
  matrix = CompatibilityMatrix(SemanticsManager())
  with patch("ml_switcheroo.cli.matrix.get_framework_priority_order", return_value=["a", "b"]):
    assert matrix._get_sorted_engines() == ["a", "b"]
