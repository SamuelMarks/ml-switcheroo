"""Test suite for the Matrix Cmd module."""

from unittest.mock import patch
from ml_switcheroo.cli.matrix import CompatibilityMatrix


def test_compatibility_matrix():
  """Verifies the behavior of compatibility matrix."""
  with patch("ml_switcheroo.cli.matrix.SemanticsManager") as MockSemantics:
    semantics = MockSemantics()
    semantics.get_known_apis.return_value = {
      "op1": {"variants": {"torch": {"api": "foo"}, "jax": {"requires_plugin": "foo"}}},
      "op2": {"variants": {"torch": None}},
    }
    semantics._key_origins = {"op1": "custom"}
    matrix = CompatibilityMatrix(semantics)
    with patch("ml_switcheroo.cli.matrix.get_framework_priority_order", return_value=["torch", "jax"]):
      res = matrix.get_json()
      assert len(res) == 2
      matrix.render()
      assert matrix._get_status_icon(None) == "❌"
      assert matrix._get_status_icon({"requires_plugin": "yes"}) == "🧩"
      assert matrix._get_status_icon({"api": "foo"}) == "✅"
