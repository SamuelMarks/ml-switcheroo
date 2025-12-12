"""
Tests for Distributed Semantics Loading (Recursion & Extensions).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


@pytest.fixture
def mock_semantics_tree(tmp_path):
  """
  Creates a mock directory structure for semantics.
  """
  # 1. Base Array API
  array_content = {"abs": {"description": "Math Abs", "variants": {"torch": {"api": "torch.abs"}}}}
  (tmp_path / "k_array_api.json").write_text(json.dumps(array_content))

  # 2. Base Neural
  neural_content = {"Linear": {"description": "Standard Linear", "variants": {"torch": {"api": "torch.nn.Linear"}}}}
  (tmp_path / "k_neural_net.json").write_text(json.dumps(neural_content))

  # 3. Extension (XGBoost)
  ext_dir = tmp_path / "extensions"
  ext_dir.mkdir()

  xgb_content = {
    "__frameworks__": {"xgboost": {"alias": {"module": "xgboost", "name": "xgb"}}},
    "XGBClassifier": {"description": "Boosted Trees", "variants": {"xgboost": {"api": "xgboost.XGBClassifier"}}},
  }
  (ext_dir / "xgboost_maps.json").write_text(json.dumps(xgb_content))

  # 4. Patch (Override Neural)
  patch_dir = ext_dir / "patches"
  patch_dir.mkdir()

  patch_content = {"Linear": {"description": "Patched Linear", "variants": {"custom": {"api": "mylib.Linear"}}}}
  (patch_dir / "neural_patch.json").write_text(json.dumps(patch_content))

  return tmp_path


def test_recursive_discovery(mock_semantics_tree):
  """
  Verify that files in root, subfolders, and nested subfolders are all loaded.
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=mock_semantics_tree):
    # Prevent loading default templates to simplify assertion
    with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
      mgr = SemanticsManager()
      mgr._reverse_index = {}

      # Check Root file load
      assert "abs" in mgr.data

      # Check Extension subfolder load
      assert "XGBClassifier" in mgr.data

      # Check Nested Patch load
      assert "Linear" in mgr.data


def test_tier_priority_override(mock_semantics_tree):
  """
  Verify loading order ensures Extensions (Extras) override Base defs.
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=mock_semantics_tree):
    mgr = SemanticsManager()
    mgr._reverse_index = {}

    # neural_patch defined: "Patched Linear" (Loaded last as Extras)
    assert mgr.data["Linear"]["description"] == "Patched Linear"

    assert "custom" in mgr.data["Linear"]["variants"]


def test_framework_config_merging(mock_semantics_tree):
  """
  Verify __frameworks__ block from extension file is merged into manager config.
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=mock_semantics_tree):
    mgr = SemanticsManager()
    mgr._reverse_index = {}

    assert "xgboost" in mgr.framework_configs
    assert mgr.framework_configs["xgboost"]["alias"]["name"] == "xgb"


def test_test_templates_via_overlay(tmp_path):
  """
  Verify that templates are loaded from overlays (snapshots).
  """
  snap = tmp_path / "snapshots"
  snap.mkdir()

  tmpl_content = {"__framework__": "custom_fw", "templates": {"import": "import custom"}}
  (snap / "custom_fw_vlatest_map.json").write_text(json.dumps(tmpl_content))

  # Need to mock resolve_snapshots_dir
  with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap):
    # Dummy semantics dir
    with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path / "semantics"):
      mgr = SemanticsManager()
      mgr._reverse_index = {}

      assert "custom_fw" in mgr.test_templates
      assert mgr.test_templates["custom_fw"]["import"] == "import custom"
