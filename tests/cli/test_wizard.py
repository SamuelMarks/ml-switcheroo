"""
Tests for CLI Wizard logic (Standard Flow).
"""

import pytest

try:
  from unittest.mock import MagicMock, patch
except ImportError:
  from mock import MagicMock, patch

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
  with patch("ml_switcheroo.cli.wizard.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()
    # Ensure we don't actually load plugins/files that might not exist
    mgr.data = {}
    w = MappingWizard(mgr)
    yield w


def test_wizard_save_logic(wizard, tmp_path):
  """
  Scenario: User selects [M]ath for `new_math_op`.
  Expect: Entry written to k_array_api.json.
  """
  api_path = "pkg.new_math_op"
  details = {"doc_summary": "Docs", "detected_sig": ["a"]}

  # Simulate saving 'math'
  wizard._save_entry(api_path, details, "k_array_api.json")

  target_file = tmp_path / "k_array_api.json"
  assert target_file.exists()
  content = target_file.read_text()
  assert "new_math_op" in content
  assert "pkg.new_math_op" in content


def test_wizard_full_flow(wizard, tmp_path):
  """
  Scenario: Run wizard against mock package.
  Inputs:
    1. new_math_op -> 'm' (Math) -> Rename 'x' -> Confirm Map Target (N)
    2. Layer -> 'n' (Neural) -> Rename 'x' -> Confirm Map Target (N)
    3. ignored -> 's' (Skip)
  Expect:
    - k_array_api.json has new_math_op
    - k_neural_net.json has Layer
    - 'ignored' is not saved anywhere
  """
  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=MockInspector()):
    # Prompt sequence (Text Inputs):
    # 1. Tier: 'm'
    # 2. Rename x: 'x' (Keep default)
    # 3. Tier: 'n'
    # 4. Rename x: 'x'
    # 5. Tier: 's'
    prompt_responses = ["m", "x", "n", "x", "s"]

    # Confirm sequence (Boolean Inputs for "Map Target?"):
    # 1. False (for math op)
    # 2. False (for layer)
    confirm_responses = [False, False]

    with patch("rich.prompt.Prompt.ask", side_effect=prompt_responses):
      with patch("rich.prompt.Confirm.ask", side_effect=confirm_responses):
        wizard.start("pkg")

  assert (tmp_path / "k_array_api.json").exists()
  assert (tmp_path / "k_neural_net.json").exists()

  # Check contents
  array_content = (tmp_path / "k_array_api.json").read_text()
  assert "new_math_op" in array_content

  neural_content = (tmp_path / "k_neural_net.json").read_text()
  assert "Layer" in neural_content


def test_resolve_target_file(wizard):
  assert wizard._resolve_target_file("math") == "k_array_api.json"
  assert wizard._resolve_target_file("neural") == "k_neural_net.json"
  assert wizard._resolve_target_file("extras") == "k_framework_extras.json"


def test_empty_scan_exits_gracefully(wizard):
  mock_inspector = MagicMock()
  mock_inspector.inspect.return_value = {}  # Empty

  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=mock_inspector):
    # Should print "No missing mappings" and return
    wizard.start("pkg")
    # No error raised
