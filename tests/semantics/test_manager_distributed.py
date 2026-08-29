"""Test suite for the Manager Distributed module."""

import json
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import patch

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics_tree(tmp_path: Path) -> Generator[Path, None, None]:
  """Provides a mock semantics tree for testing.

  Args:
      tmp_path (Path): Tmp path pytest fixture.

  Yields:
      Path: Tmp path.
  """
  array_content: Dict[str, Dict[str, Any]] = {
    "abs": {"description": "Math Abs", "variants": {"torch": {"api": "torch.abs"}}}
  }
  import yaml

  array_content["abs"]["operation"] = "abs"
  (tmp_path / "k_array_api.yaml").write_text(yaml.dump(array_content["abs"]))
  neural_content: Dict[str, Dict[str, Any]] = {
    "Linear": {"description": "Standard Linear", "variants": {"torch": {"api": "torch.nn.Linear"}}}
  }
  neural_content["Linear"]["operation"] = "Linear"
  (tmp_path / "k_neural_net.yaml").write_text(yaml.dump(neural_content["Linear"]))
  ext_dir: Path = tmp_path / "extensions"
  ext_dir.mkdir()
  xgb_content: Dict[str, Dict[str, Any]] = {
    "__frameworks__": {"xgboost": {"alias": {"module": "xgboost", "name": "xgb"}}},
    "XGBClassifier": {"description": "Boosted Trees", "variants": {"xgboost": {"api": "xgboost.XGBClassifier"}}},
  }
  (ext_dir / "xgboost_maps.yaml").write_text(yaml.dump(xgb_content))
  patch_dir: Path = ext_dir / "patches"
  patch_dir.mkdir()
  patch_content: Dict[str, Dict[str, Any]] = {
    "Linear": {"description": "Patched Linear", "variants": {"custom": {"api": "mylib.Linear"}}}
  }
  patch_content["Linear"]["operation"] = "Linear"
  (patch_dir / "neural_patch.yaml").write_text(yaml.dump(patch_content["Linear"]))
  yield tmp_path


def test_recursive_discovery(mock_semantics_tree: Path) -> None:
  """Verifies the behavior of recursive discovery.

  Args:
      mock_semantics_tree (Path): Mock tree path.
  """
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=mock_semantics_tree):
    with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
      mgr: SemanticsManager = SemanticsManager()
      mgr._reverse_index = {}
      assert "abs" in mgr.data
      assert "XGBClassifier" in mgr.data
      assert "Linear" in mgr.data


def test_tier_priority_override(mock_semantics_tree: Path) -> None:
  """Verifies the behavior of tier priority override.

  Args:
      mock_semantics_tree (Path): Mock tree path.
  """
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=mock_semantics_tree):
    with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
      mgr: SemanticsManager = SemanticsManager()
      mgr._reverse_index = {}
      assert mgr.data["Linear"]["description"] == "Patched Linear"
      assert "custom" in mgr.data["Linear"]["variants"]


def test_framework_config_merging(mock_semantics_tree: Path) -> None:
  """Verifies the behavior of framework configuration merging.

  Args:
      mock_semantics_tree (Path): Mock tree path.
  """
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=mock_semantics_tree):
    with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
      mgr: SemanticsManager = SemanticsManager()
      mgr._reverse_index = {}
      assert "xgboost" in mgr.framework_configs
      assert mgr.framework_configs["xgboost"]["alias"]["name"] == "xgb"


def test_test_templates_via_overlay(tmp_path: Path) -> None:
  """Tests templates via overlay.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  snap: Path = tmp_path / "snapshots"
  snap.mkdir()
  sem: Path = tmp_path / "semantics"
  sem.mkdir()
  tmpl_content: Dict[str, Any] = {"__framework__": "custom_fw", "templates": {"import": "import custom"}}
  (snap / "custom_fw_vlatest_map.json").write_text(json.dumps(tmpl_content))
  with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap):
    with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem):
      with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
        mgr: SemanticsManager = SemanticsManager()
        mgr._reverse_index = {}
        assert "custom_fw" in mgr.test_templates
        assert mgr.test_templates["custom_fw"]["import"] == "import custom"
