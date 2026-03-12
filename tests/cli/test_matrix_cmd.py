import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.matrix import CompatibilityMatrix


def test_compatibility_matrix():
  with patch("ml_switcheroo.cli.matrix.SemanticsManager") as MockSemantics:
    semantics = MockSemantics()

    # provide get_known_apis data
    semantics.get_known_apis.return_value = {
      "op1": {"variants": {"torch": {"api": "foo"}, "jax": {"requires_plugin": "foo"}}},
      "op2": {"variants": {"torch": None}},
    }

    # provide _key_origins
    semantics._key_origins = {"op1": "custom"}

    matrix = CompatibilityMatrix(semantics)

    with patch("ml_switcheroo.cli.matrix.get_framework_priority_order", return_value=["torch", "jax"]):
      res = matrix.get_json()
      assert len(res) == 2

      # Check cell generation
      matrix.render()

      # test line 149
      assert matrix._get_status_icon(None) == "❌"
      # test line 153
      assert matrix._get_status_icon({"requires_plugin": "yes"}) == "🧩"
      # test line 156
      assert matrix._get_status_icon({"api": "foo"}) == "✅"
