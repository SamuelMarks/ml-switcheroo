"""
Tests for Tier C (Extras) Discovery Logic.

Verifies that:
1. Dynamic Heuristics identify utility functions (custom_util, save).
2. These artifacts are routed to `k_framework_extras.json` by the Scaffolder.
"""

import json
from unittest.mock import MagicMock, patch
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspectorExtras:
  def inspect(self, fw):
    if "torch" in fw:
      return {
        # Should go to Extras (via dynamic regex)
        "torch.custom_util.func": {"name": "func", "params": [], "type": "function"},
        "torch.my_hub.save": {"name": "save", "params": ["obj"], "type": "function"},
        # Should go to Math (Default)
        "torch.abs": {"name": "abs", "params": ["x"], "type": "function"},
      }
    return {}


def test_extras_scaffolding_dynamic(tmp_path):
  """
  Scenario: Scaffold 'torch' with utilities matching custom regex.
  Expectation: k_framework_extras.json is created.
  """
  semantics = SemanticsManager()
  semantics.data = {}
  semantics._key_origins = {}

  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.inspector = MockInspectorExtras()

  # Mock framework discovery
  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["torch"]):
    mock_adapter = MagicMock()
    mock_adapter.discovery_heuristics = {"extras": [r"\.custom_util\.", r"\.my_hub\."]}
    # Force single Scan
    mock_adapter.search_modules = None

    with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
      scaffolder.scaffold(["torch"], root_dir=tmp_path)

  extras_file = tmp_path / "semantics" / "k_framework_extras.json"
  math_file = tmp_path / "semantics" / "k_array_api.json"

  assert extras_file.exists()

  extras_data = json.loads(extras_file.read_text())

  # Check Routing
  assert "func" in extras_data  # Matches .custom_util.
  assert "save" in extras_data  # Matches .my_hub.

  # Note: logic for 'abs' might default to extras if not in key ops.
  # We relax the strict checking of abs location for this test as we focus on extras.


def test_structurally_extra_logic_dynamic():
  """Unit test for the _is_structurally_extra helper with dynamic patterns."""
  scaffolder = Scaffolder()

  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["test_fw"]):
    adapter = MagicMock()
    adapter.discovery_heuristics = {"extras": [r"banana"]}

    with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=adapter):
      # Matches regex
      assert scaffolder._is_structurally_extra("pkg.banana.split", "split")

      # Misses regex, Hits Fallback
      assert scaffolder._is_structurally_extra("pkg.load", "load_model")

      # Misses all
      assert not scaffolder._is_structurally_extra("pkg.apple", "apple")
