"""
Integration Tests for CLI Trace dumping (Feature 05).

Verifies that:
1. `convert --json-trace` produces a valid JSON file.
2. content includes AST Mutation snapshots.
"""

import json
import pytest
from typing import Set
from unittest.mock import patch
from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.semantics.manager import SemanticsManager


class MockTraceSemantics(SemanticsManager):
  """Mock semantics to ensure torch.abs is rewritten."""

  def __init__(self):
    # Skip file loading
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # Inject torch.abs definition
    self.data["abs"] = {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}, "std_args": ["x"]}
    self._reverse_index["torch.abs"] = ("abs", self.data["abs"])

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods


def test_cli_trace_output_generation(tmp_path, capsys):
  """
  Scenario: Convert file with --json-trace flag.
  Expectation: trace JSON file created containing events.
  """
  infile = tmp_path / "input.py"
  infile.write_text("y = torch.abs(x)")

  outfile = tmp_path / "output.py"
  tracefile = tmp_path / "trace.json"

  args = [
    "convert",
    str(infile),
    "--out",
    str(outfile),
    "--source",
    "torch",
    "--target",
    "jax",
    "--json-trace",
    str(tracefile),
  ]

  # Patch the SemanticsManager used in the CLI commands module
  with patch("ml_switcheroo.cli.commands.SemanticsManager", MockTraceSemantics):
    try:
      main(args)
    except SystemExit as e:
      # We expect exit code 0
      if e.code != 0:
        raise e

  # Verify Trace File Exists
  assert tracefile.exists(), "Trace file was not created"

  # Load and Inspect
  content = tracefile.read_text()
  try:
    data = json.loads(content)
  except json.JSONDecodeError:
    pytest.fail(f"Invalid JSON in trace file: {content}")

  assert isinstance(data, list)
  assert len(data) > 0

  # Identify key event types
  event_types = {e["type"] for e in data}
  assert "phase_start" in event_types
  assert "ast_mutation" in event_types

  # Check Code Diff Capture
  # Filter for the relevant logic change
  mutation = next(
    (e for e in data if e["type"] == "ast_mutation" and "torch.abs" in e["metadata"].get("before", "")), None
  )

  assert mutation is not None, (
    f"Could not find mutation event for 'torch.abs'. Events found: {[e['description'] for e in data if e['type'] == 'ast_mutation']}"
  )
  assert "after" in mutation["metadata"]
  # Verify the transformation logic occurred in the trace
  assert "jax.numpy.abs" in mutation["metadata"]["after"]
