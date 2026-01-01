"""
Tests for CLI 'Convert' Command with Intermediate Representation Flag.

Verifies that:
1. --intermediate flag is accepted.
2. Passing 'mlir' triggers the MLIR bridge in the trace logic.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.semantics.manager import SemanticsManager


class MockTraceSemantics(SemanticsManager):
  """Mock semantics to ensure torch.abs is processed and traced."""

  def __init__(self):
    # Skip file loading
    self.data = {}
    self._providers = {}
    self._source_registry = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()
    self.test_templates = {}

    # Inject minimal definition
    self.data["abs"] = {
      "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}},
      "std_args": ["x"],
    }
    self._reverse_index["torch.abs"] = ("abs", self.data["abs"])

  def get_all_rng_methods(self):
    return set()

  def get_framework_config(self, fw):
    return {}


def test_cli_intermediate_flag_mlir(tmp_path):
  """
  Scenario: User runs `convert ... --intermediate mlir --json-trace trace.json`.
  Expectation: Trace JSON contains "MLIR Bridge" events.
  """
  infile = tmp_path / "model.py"
  infile.write_text("x = torch.abs(y)")
  outfile = tmp_path / "converted.py"
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
    "--intermediate",
    "mlir",
    "--json-trace",
    str(tracefile),
  ]

  # Patch SemanticsManager
  with patch("ml_switcheroo.cli.handlers.convert.SemanticsManager", MockTraceSemantics):
    try:
      main(args)
    except SystemExit as e:
      assert e.code == 0, "CLI exited with error"

  assert outfile.exists()
  assert tracefile.exists()

  # Analyze Trace
  trace_data = json.loads(tracefile.read_text())

  # Check for "MLIR Bridge" phase start
  mlir_phases = [e for e in trace_data if e["type"] == "phase_start" and "MLIR Bridge" in e["description"]]
  assert len(mlir_phases) > 0, "MLIR Bridge phase not found in trace"

  # Check for "MLIR Generation" mutation
  mlir_mutation = [e for e in trace_data if e["type"] == "ast_mutation" and "MLIR Generation" in e["description"]]
  assert len(mlir_mutation) > 0, "MLIR Generation mutation not found in trace"

  # Verify generated code is still valid python (re-rolled)
  code = outfile.read_text()
  assert "jax.numpy.abs" in code
