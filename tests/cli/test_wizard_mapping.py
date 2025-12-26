"""
Tests for Wizard Mapping Logic (Including Plugin Support).

Verifies that:
1. Users can rename detected arguments to Standard names.
2. Source mapping ({std: source}) is generated correctly.
3. Target mapping (JAX) can be defined inline.
4. **Plugin assignments** are correctly prompted and saved.
"""

import json
import pytest

from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.wizard import MappingWizard
from ml_switcheroo.semantics.manager import SemanticsManager

# --- Mocks ---


class MockInspector:
  def inspect(self, _pkg, **kwargs):
    return {
      "torch.full_op": {
        "name": "full_op",
        "params": ["input", "dim"],  # Detects 'input', 'dim'
        "docstring_summary": "Full Logic Op",
      }
    }


@pytest.fixture
def wizard(tmp_path):
  # Patch file resolution to use tmp_path
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"

  # Ensure Dirs Mocked
  sem_dir.mkdir()
  snap_dir.mkdir()

  # FIX: Patch where it is used (cli.wizard)
  with patch("ml_switcheroo.cli.wizard.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.cli.wizard.resolve_snapshots_dir", return_value=snap_dir):
      mgr = SemanticsManager()
      mgr._reverse_index = {}
      w = MappingWizard(mgr)
      mgr.data = {}
      mgr._reverse_index = {}
      yield w


def test_arg_normalization_flow(wizard, tmp_path):
  """
  Scenario:
      Detected: torch.full_op(input, dim)
      User Action:
          - Bucket: [M]ath
          - Arg 'input': Rename to 'x'
          - Arg 'dim': Rename to 'axis'
          - Map Target? No
  Expectation:
      k_array_api.json entry:
      std_args: ["x", "axis"]
      variants.torch: { "api": "...", "args": {"x": "input", "axis": "dim"} }
  """
  # Prompt Sequence (Strings):
  # 1. Bucket -> 'm'
  # 2. Rename 'input' -> 'x'
  # 3. Rename 'dim' -> 'axis'
  prompts = ["m", "x", "axis"]

  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=MockInspector()):
    with patch("rich.prompt.Prompt.ask", side_effect=prompts):
      with patch("rich.prompt.Confirm.ask", return_value=False):  # No target map
        wizard.start("torch")

  # Verify Spec JSON
  fpath = tmp_path / "semantics" / "k_array_api.json"
  assert fpath.exists()

  data = json.loads(fpath.read_text())
  assert "full_op" in data
  entry = data["full_op"]

  # Check Standards
  assert entry["std_args"] == ["x", "axis"]

  # Verify Snapshot Mapping
  snap_path = tmp_path / "snapshots" / "torch_vlatest_map.json"
  assert snap_path.exists()
  snap_data = json.loads(snap_path.read_text())

  torch_var = snap_data["mappings"]["full_op"]
  assert torch_var["api"] == "torch.full_op"
  assert torch_var["args"] == {"x": "input", "axis": "dim"}


def test_full_bidirectional_flow(wizard, tmp_path):
  """
  Scenario:
      Detected: torch.full_op(input)
      User Action:
          ...
          - Map Target? Yes
          - Target FW: 'jax'
          - Target API: 'jax.numpy.op'
          - **Plugin? No**
          - Tgt Arg 'x': 'a'
          ...
  """
  # Mock has params=["input", "dim"]

  # Prompt sequence (Text):
  # 1. Tier: 'm'
  # 2. Source norm 1: 'x'
  # 3. Source norm 2: 'axis'
  # 4. Tgt FW: 'jax'
  # 5. Tgt API: 'jax.numpy.op'
  # 6. Tgt map 1: 'a'
  # 7. Tgt map 2: 'axis'
  prompt_side_effect = [
    "m",  # Tier
    "x",  # Source norm 1
    "axis",  # Source norm 2
    "jax",  # Tgt FW
    "jax.numpy.op",  # Tgt API
    "a",  # Tgt map 1
    "axis",  # Tgt map 2
  ]

  # Confirm sequence (Bool):
  # 1. Map Target? -> True
  # 2. Plugin?     -> False
  confirm_side_effect = [True, False]

  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=MockInspector()):
    with patch("rich.prompt.Prompt.ask", side_effect=prompt_side_effect):
      with patch("rich.prompt.Confirm.ask", side_effect=confirm_side_effect):
        wizard.start("torch")

  # Check JAX Overlay exists
  jax_file = tmp_path / "snapshots" / "jax_vlatest_map.json"
  assert jax_file.exists()

  data = json.loads(jax_file.read_text())
  jax_var = data["mappings"]["full_op"]

  assert jax_var["api"] == "jax.numpy.op"
  assert jax_var.get("requires_plugin") is None  # Should be absent
  assert jax_var["args"] == {"x": "a"}


def test_wizard_plugin_assignment(wizard, tmp_path):
  """
  Scenario:
      Detected: torch.full_op(input, dim)
      User Action:
          ...
          - Map Target? Yes
          - Target FW: 'jax'
          - Target API: 'jax.numpy.op'
          - **Plugin? Yes**
          - **Plugin Name: 'decompose_alpha'**
          ...
  Expectation:
      JSON variant for JAX contains "requires_plugin": "decompose_alpha"
  """
  # Prompt sequence:
  # 1. m
  # 2. x
  # 3. axis
  # 4. jax
  # 5. jax.numpy.op
  # 6. decompose_alpha  <-- New Prompt because Plugin=True
  # 7. a
  # 8. axis
  prompt_side_effect = [
    "m",
    "x",
    "axis",
    "jax",
    "jax.numpy.op",
    "decompose_alpha",  # Plugin name provided
    "a",
    "axis",
  ]

  # Confirm sequence:
  # 1. Map Target? -> True
  # 2. Plugin?     -> True
  confirm_side_effect = [True, True]

  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=MockInspector()):
    with patch("rich.prompt.Prompt.ask", side_effect=prompt_side_effect):
      with patch("rich.prompt.Confirm.ask", side_effect=confirm_side_effect):
        wizard.start("torch")

  jax_file = tmp_path / "snapshots" / "jax_vlatest_map.json"
  data = json.loads(jax_file.read_text())
  jax_var = data["mappings"]["full_op"]

  assert jax_var.get("requires_plugin") == "decompose_alpha"


def test_skipping_logic_preserved(wizard, tmp_path):
  """Ensure standard skip flow still works."""
  with patch("ml_switcheroo.cli.wizard.ApiInspector", return_value=MockInspector()):
    with patch("rich.prompt.Prompt.ask", return_value="s"):  # Skip
      wizard.start("torch")

  # No Spec file
  assert not (tmp_path / "semantics" / "k_array_api.json").exists()
