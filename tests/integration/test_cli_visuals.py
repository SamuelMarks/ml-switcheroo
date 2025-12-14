"""
Integration Visual Tests for CLI Output.
Ensures that the Rich table formatting and JSON reports remain stable.
"""

import json
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager
from rich.console import Console


# Use a mock Semantics Manager to ensure snapshot stability.
class StableMockSemantics(SemanticsManager):
  def get_known_apis(self):
    return {
      "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "magic_op": {
        "std_args": ["x"],
        "variants": {"torch": {"api": "torch.magic"}, "jax": {"requires_plugin": "magic_fix"}},
      },
      "unsupported_op": {
        "std_args": ["x"],
        "variants": {
          "torch": {"api": "torch.oops"}
          # No JAX
        },
      },
    }

  def get_definition(self, api_name):
    if api_name == "torch.abs":
      return "abs", {}
    return None


class MockInspector:
  def inspect(self, _pkg):
    return {
      "torch.abs": {"name": "abs", "params": ["x"], "docstring_summary": "Calculates abs."},
      "torch.new_thing": {"name": "new_thing", "params": ["a", "b"], "docstring_summary": "Brand new feature."},
    }


def test_matrix_visual_snapshot(snapshot, tmp_path):
  """
  Verifies the ASCII output of the Compatibility Matrix table.
  Captures Rich Console output into a string.
  """
  semantics = StableMockSemantics()

  # Configure console to capture string, force valid width for consistent wrap
  console = Console(file=None, force_terminal=True, width=100, record=True)

  matrix = CompatibilityMatrix(semantics)
  matrix.console = console  # Inject capture console

  # NOTE: Patch 'ml_switcheroo.config' since 'cli.matrix' imports from there.
  # This ensures we control the columns. Removed Keras to match snapshot.
  mock_fws = ["torch", "jax", "numpy", "tensorflow", "mlx", "paxml"]

  # FIX: We need to ensure that get_adapter returns mocks with 'ui_priority' attributes
  # to match the sort order defined in the snapshot (Torch=0, Jax=10, etc.)
  def mock_adapter_factory(name):
    m = MagicMock()
    priorities = {"torch": 0, "jax": 10, "numpy": 20, "tensorflow": 30, "mlx": 50, "paxml": 60}
    m.ui_priority = priorities.get(name, 999)
    return m

  with patch("ml_switcheroo.config.available_frameworks", return_value=mock_fws):
    # Patched updated location
    with patch("ml_switcheroo.frameworks.get_adapter", side_effect=mock_adapter_factory):
      matrix.render()

  output = console.export_text()

  def header_insensitive(text: str) -> str:
    lines = text.splitlines()
    if not lines:
      return text
    lines[0] = lines[0].strip()
    return "\n".join(lines) + "\n"

  snapshot.assert_match(output, normalizer=header_insensitive)
