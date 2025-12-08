"""
Tests for Tier C (Extras) Discovery Logic.

Verifies that:
1. Heuristics correctly identify utility functions (save, load, seed, no_grad).
2. These artifacts are routed to `k_framework_extras.json` by the Scaffolder.
3. Logic works even with empty Semantics (bootstrapping mode).
"""

import json
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspectorExtras:
  def inspect(self, fw):
    if fw == "torch":
      return {
        # Should go to Extras (Keyword Heuristic)
        "torch.manual_seed": {"name": "manual_seed", "params": ["seed"], "type": "function"},
        "torch.save": {"name": "save", "params": ["obj", "f"], "type": "function"},
        # Should go to Extras (Submodule Heuristic)
        "torch.utils.data.DataLoader": {"name": "DataLoader", "params": ["dataset"], "type": "class"},
        # Should go to Extras (Context/Grad Heuristic)
        "torch.no_grad": {"name": "no_grad", "params": [], "type": "class"},
        # Should go to Math (Default)
        "torch.abs": {"name": "abs", "params": ["x"], "type": "function"},
      }
    return {}


def test_extras_scaffolding_writes_to_correct_file(tmp_path):
  """
  Scenario: Scaffold 'torch' with utilities.
  Expectation: k_framework_extras.json is created and populated.
  """
  # Use empty semantics to force heuristic path
  semantics = SemanticsManager()
  semantics.data = {}

  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.inspector = MockInspectorExtras()

  scaffolder.scaffold(["torch"], tmp_path)

  extras_file = tmp_path / "k_framework_extras.json"
  math_file = tmp_path / "k_array_api.json"

  assert extras_file.exists(), "Extras file not created"
  assert math_file.exists(), "Math file not created"

  extras_data = json.loads(extras_file.read_text())
  math_data = json.loads(math_file.read_text())

  # Check Routing
  assert "manual_seed" in extras_data
  assert "save" in extras_data
  assert "DataLoader" in extras_data
  assert "no_grad" in extras_data

  # Check Isolation
  assert "abs" in math_data
  assert "abs" not in extras_data
  assert "manual_seed" not in math_data


def test_structurally_extra_logic():
  """Unit test for the _is_structurally_extra helper."""
  scaffolder = Scaffolder()

  # Keywords
  assert scaffolder._is_structurally_extra("pkg.manual_seed", "manual_seed")
  assert scaffolder._is_structurally_extra("pkg.load", "load_model")
  assert scaffolder._is_structurally_extra("pkg.device", "device")

  # Contexts
  assert scaffolder._is_structurally_extra("pkg.no_grad", "no_grad")
  assert scaffolder._is_structurally_extra("pkg.set_grad_enabled", "set_grad_enabled")

  # Submodules
  assert scaffolder._is_structurally_extra("pkg.utils.data.stuff", "stuff")
  assert scaffolder._is_structurally_extra("pkg.distributed.init", "init")

  # Negatives (Math/Neural)
  assert not scaffolder._is_structurally_extra("pkg.abs", "abs")
  assert not scaffolder._is_structurally_extra("pkg.nn.Linear", "Linear")
  assert not scaffolder._is_structurally_extra("pkg.conv2d", "conv2d")
