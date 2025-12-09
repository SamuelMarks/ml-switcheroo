"""
Tests for Distributed Semantics Loading (Recursion & Extensions).

Verifies that:
1.  Recursively finds JSON files in subdirectories.
2.  Prioritizes loading order (Array -> Neural -> Extras).
3.  Allows 'Extension Fragment' files to patch or add definitions.
4.  Merges `__frameworks__` config from extensions correctly.
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

  root/
    k_array_api.json       (Base math)
    k_neural_net.json      (Layers)
    extensions/
      xgboost_maps.json    (New framework extension)
      patches/
        neural_patch.json  (Overrides)
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
  # Should load LAST due to 'Extras' priority logic or file location if named correctly?
  # By default, files without 'array'/'neural' in name are 'Extras' priority (highest precedence).
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

      # Check Root file load
      assert "abs" in mgr.data

      # Check Extension subfolder load
      assert "XGBClassifier" in mgr.data

      # Check Nested Patch load
      # 'Linear' is defined in k_neural_net, but patched in extensions/patches/neural_patch.json
      assert "Linear" in mgr.data


def test_tier_priority_override(mock_semantics_tree):
  """
  Verify loading order ensures Extensions (Extras) override Base defs.

  Order:
  1. k_array_api.json (Array Tier, Priority 10)
  2. k_neural_net.json (Neural Tier, Priority 20)
  3. neural_patch.json (No keyword, defaults to Extras, Priority 30)

  Expectation: Linear description should be "Patched Linear".
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=mock_semantics_tree):
    mgr = SemanticsManager()

    # k_neural_net defined: "Standard Linear"
    # neural_patch defined: "Patched Linear"
    # Patch should win.
    assert mgr.data["Linear"]["description"] == "Patched Linear"

    # Verify variants merged/updated
    # Original had "torch". Patch added "custom".
    # Note: _merge_tier replaces the dict if key exists, does NOT deep merge operation dictionary content
    # (It checks if op_name in data, then assigns self.data[op_name] = stored_dict).
    # So "variants" should only contain "custom" unless the patch file replicated "torch".
    # Let's check implementation of _merge_tier:
    # self.data[op_name] = stored_dict (Complete replacement specific key).

    assert "custom" in mgr.data["Linear"]["variants"]
    # Torch variant is lost if patch didn't include it. This is intended "Override" behavior.
    assert "torch" not in mgr.data["Linear"]["variants"]


def test_framework_config_merging(mock_semantics_tree):
  """
  Verify __frameworks__ block from extension file is merged into manager config.
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=mock_semantics_tree):
    mgr = SemanticsManager()

    # xgboost config was in extensions/xgboost_maps.json
    assert "xgboost" in mgr.framework_configs
    assert mgr.framework_configs["xgboost"]["alias"]["name"] == "xgb"


def test_test_templates_globbing(tmp_path):
  """
  Verify that k_test_templates.json is found and loaded even recursively.
  """
  # Create template in a subfolder to test robust finding logic
  sub = tmp_path / "config"
  sub.mkdir()

  tmpl_content = {"custom_fw": {"import": "import custom"}}
  (sub / "k_test_templates.json").write_text(json.dumps(tmpl_content))

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()

    assert "custom_fw" in mgr.test_templates
    assert mgr.test_templates["custom_fw"]["import"] == "import custom"
