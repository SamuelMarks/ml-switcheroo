"""
Tests for Updater (Discovery & Merging).

Verifies:
1.  Gap detection between package scan and finding inputs.
2.  Tier suggestion heuristics (Math vs Neural).
3.  **Auto Merge**: Writing discovered APIs to JSON files on disk.
4.  Preservation of existing manual entries.
"""

import json
import pytest
from unittest.mock import patch

from ml_switcheroo.discovery.updater import MappingsUpdater
from ml_switcheroo.semantics.manager import SemanticsManager


# Mock Inspector
class MockInspector:
  def inspect(self, _pkg):
    return {
      "pkg.known_func": {"params": ["x"], "docstring_summary": "Known"},
      "pkg.new_feature": {"params": ["a", "b"], "docstring_summary": "New stuff"},
      "pkg.nn.Layer": {"params": ["x"], "docstring_summary": "Neural Layer"},
    }


# Mock Semantics
class MockSemantics(SemanticsManager):
  def __init__(self):
    # We manually init structures
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {"pkg.known_func": ("known_func", {})}

  def get_definition(self, api):
    return self._reverse_index.get(api)


@pytest.fixture
def updater():
  semantics = MockSemantics()
  upd = MappingsUpdater(semantics)
  upd.inspector = MockInspector()
  return upd


def test_find_gaps_logic(updater, tmp_path):
  """
  Verify detection of missing APIs using the update_package workflow.
  """
  report_file = tmp_path / "report.json"

  # Run in report mode
  missing = updater.update_package("pkg", auto_merge=False, report_path=report_file)

  assert "pkg.known_func" not in missing
  assert "pkg.new_feature" in missing
  assert "pkg.nn.Layer" in missing

  # Check report file logic
  assert report_file.exists()
  data = json.loads(report_file.read_text())
  assert "pkg.new_feature" in data
  assert data["pkg.new_feature"]["detected_sig"] == ["a", "b"]


def test_tier_guessing(updater):
  """Verify tier assignment heuristic."""
  assert updater._guess_tier("torch.nn.Linear") == "k_neural_net.json"
  assert updater._guess_tier("pkg.layers.Conv") == "k_neural_net.json"
  assert updater._guess_tier("torch.abs") == "k_array_api.json"
  assert updater._guess_tier("torch.Tensor") == "k_neural_net.json"  # uppercase


def test_auto_merge_writes_to_disk(updater, tmp_path):
  """
  Verify that auto_merge=True writes to the JSON files in the semantics dir.
  """
  # 1. Mock resolve_semantics_dir to point to tmp_path
  with patch("ml_switcheroo.discovery.updater.resolve_semantics_dir", return_value=tmp_path):
    # 2. Run Update
    updater.update_package("pkg", auto_merge=True)

    # 3. Verify Files Created
    neural_path = tmp_path / "k_neural_net.json"
    array_path = tmp_path / "k_array_api.json"

    assert neural_path.exists()
    assert array_path.exists()

    # 4. Verify Content
    neural_data = json.loads(neural_path.read_text())
    array_data = json.loads(array_path.read_text())

    # pkg.nn.Layer -> Neural
    assert "Layer" in neural_data
    assert neural_data["Layer"]["variants"]["pkg"]["api"] == "pkg.nn.Layer"

    # pkg.new_feature -> Array
    assert "new_feature" in array_data
    assert array_data["new_feature"]["variants"]["pkg"]["api"] == "pkg.new_feature"


def test_rich_reporting_no_crash(updater):
  """Ensure Rich table printing executes without error."""
  updater.update_package("pkg", auto_merge=False)
  # If we got here, no crash in table generation.


def test_merge_preserves_manual_entries(updater, tmp_path):
  """
  Ensure existing keys in JSON are not overwritten by automated discovery.
  """
  with patch("ml_switcheroo.discovery.updater.resolve_semantics_dir", return_value=tmp_path):
    # 1. Create existing file with manual data
    array_path = tmp_path / "k_array_api.json"
    array_path.write_text(json.dumps({"new_feature": {"description": "Manual Override"}}))

    # 2. Run Update (pkg.new_feature would normally map to 'new_feature')
    updater.update_package("pkg", auto_merge=True)

    # 3. Verify preservation
    data = json.loads(array_path.read_text())
    assert data["new_feature"]["description"] == "Manual Override"
