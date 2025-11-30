"""
Integration Visual Tests for CLI Output.
Ensures that the Rich table formatting and JSON reports remain stable.
"""

import json
from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.discovery.updater import MappingsUpdater
from rich.console import Console


# Use a mock Semantics Manager to ensure snapshot stability.
# Real data changes too often to snapshot directly.
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

  # Needs reverse index logic for Updater test
  def get_definition(self, api_name):
    # We simulate that 'abs' is known, but 'new_thing' is missing
    if api_name == "torch.abs":
      return "abs", {}
    return None


class MockInspector:
  """Deterministic inspector."""

  def inspect(self, _pkg):
    return {
      "torch.abs": {"name": "abs", "params": ["x"], "docstring_summary": "Calculates abs."},
      "torch.new_thing": {"name": "new_thing", "params": ["a", "b"], "docstring_summary": "Brand new feature."},
    }


def test_matrix_visual_snapshot(snapshot):
  """
  Verifies the ASCII output of the Compatibility Matrix table.
  Captures Rich Console output into a string.
  """
  semantics = StableMockSemantics()

  # Configure console to capture string, force valid width for consistent wrap
  console = Console(file=None, force_terminal=True, width=100, record=True)

  matrix = CompatibilityMatrix(semantics)
  matrix.console = console  # Inject capture console

  matrix.render()

  # Export text (handling ANSI codes usually stripped for snapshot readability,
  # or kept if we want to test colors. text=True strips style).
  output = console.export_text()

  # Helper to ignore centering whitespace on the title line (line 0)
  def header_insensitive(text: str) -> str:
    lines = text.splitlines()
    if not lines:
      return text
    # ID matching: Strip whitespace from the title line only
    lines[0] = lines[0].strip()
    # Rejoin with standard newline to ensure cross-platform check matches
    return "\n".join(lines) + "\n"

  snapshot.assert_match(output, normalizer=header_insensitive)


def test_update_report_json_snapshot(snapshot, tmp_path):
  """
  Verifies the 'missing_mappings.json' report structure correctly identifies
  missing APIs and formats the suggestion dict properly.
  """
  semantics = StableMockSemantics()
  updater = MappingsUpdater(semantics)
  updater.inspector = MockInspector()

  report_path = tmp_path / "report_snapshot.json"

  # Logic: inspects mock 'torch', diffs against mock semantics.
  # torch.abs is known -> ignored.
  # torch.new_thing is unknown -> included.
  # We use update_package in report-only mode
  updater.update_package("torch", auto_merge=False, report_path=report_path)

  # Read the generated JSON, then format it deterministically
  with open(report_path, "rt", encoding="utf-8") as f:
    data = json.load(f)

  formatted_json = json.dumps(data, indent=2, sort_keys=True)
  snapshot.assert_match(formatted_json, extension="json")
