"""
Tests for CLI Wizard logic (Standard Flow).
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.wizard import MappingWizard
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector:
  def inspect(self, _pkg):
    return {
      "pkg.new_math_op": {"params": ["x"]},
      "pkg.nn.Layer": {"params": ["x"]},
      "pkg.ignored": {"params": ["x"]},
    }


@pytest.fixture
def wizard(tmp_path):
  # Patch file resolution to use tmp_path
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"

  with patch("ml_switcheroo.cli.wizard.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.cli.wizard.resolve_snapshots_dir", return_value=snap_dir):
      mgr = SemanticsManager()
      mgr._reverse_index = {}
      mgr.data = {}
      w = MappingWizard(mgr)
      mgr.data = {}
      mgr._reverse_index = {}
      yield w


def test_wizard_save_logic(wizard, tmp_path):
  api_path = "pkg.new_math_op"
  details = {"doc_summary": "Docs", "detected_sig": ["a"]}

  wizard._save_complex_entry(
    filename="k_array_api.json",
    api_path=api_path,
    doc_summary="Docs",
    std_args=["a"],
    source_fw="pkg",
    source_arg_map={},
    target_variant=None,
  )

  # Check Spec
  target_file = tmp_path / "semantics" / "k_array_api.json"
  assert target_file.exists()
  content = target_file.read_text()
  assert "new_math_op" in content
  assert "variants" not in content

  # Check Mapping
  map_file = tmp_path / "snapshots" / "pkg_vlatest_map.json"
  assert map_file.exists()
  map_data = json.loads(map_file.read_text())
  assert map_data["mappings"]["new_math_op"]["api"] == "pkg.new_math_op"


def test_wizard_full_flow(wizard, tmp_path):
  """
  Scenario: Run wizard against mock package.
  Items sorted by key:
  1. pkg.ignored (param x)
  2. pkg.new_math_op (param x)
  3. pkg.nn.Layer (param x)
  """
  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=MockInspector()):
    # Prompt Sequence:
    # 1. Ignored: Bucket 'm' (Math), Arg 'x' rename (keep 'x')
    # 2. NewMath: Bucket 'm' (Math), Arg 'x' rename (keep 'x')
    # 3. Layer:   Bucket 'n' (Neural), Arg 'x' rename (keep 'x')
    prompt_responses = ["m", "x", "m", "x", "n", "x"]

    # Confirm actions: Map Target? (False for all 3)
    confirm_responses = [False, False, False]

    with patch("rich.prompt.Prompt.ask", side_effect=prompt_responses):
      with patch("rich.prompt.Confirm.ask", side_effect=confirm_responses):
        wizard.start("pkg")

  # Verify Spec Files
  sem_dir = tmp_path / "semantics"
  assert (sem_dir / "k_array_api.json").exists()
  assert (sem_dir / "k_neural_net.json").exists()

  array_content = (sem_dir / "k_array_api.json").read_text()
  assert "new_math_op" in array_content
  assert "ignored" in array_content

  neural_content = (sem_dir / "k_neural_net.json").read_text()
  assert "Layer" in neural_content

  # Verify Mapping Files
  snap_dir = tmp_path / "snapshots"
  snap_file = snap_dir / "pkg_vlatest_map.json"
  assert snap_file.exists()
  mappings = json.loads(snap_file.read_text())["mappings"]
  assert "new_math_op" in mappings
  assert "Layer" in mappings


def test_resolve_target_file(wizard):
  assert wizard._resolve_target_file("math") == "k_array_api.json"
  assert wizard._resolve_target_file("neural") == "k_neural_net.json"
  assert wizard._resolve_target_file("extras") == "k_framework_extras.json"


def test_empty_scan_exits_gracefully(wizard):
  mock_inspector = MagicMock()
  mock_inspector.inspect.return_value = {}
  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=mock_inspector):
    wizard.start("pkg")
