"""
Tests for the JSON Specification Injector.

Verifies that StandardsInjector correctly loads, updates, and saves
JSON specification files in the Semantics Knowledge Base.
"""

import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

import pytest
from ml_switcheroo.core.dsl import OperationDef, OpType
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.tools.injector_spec import StandardsInjector


@pytest.fixture
def sample_op():
  return OperationDef(
    operation="LogSoftmax", description="Log Softmax.", op_type=OpType.FUNCTION, std_args=["x", "dim"], variants={}
  )


def test_injector_finds_correct_file(sample_op):
  """Verify heuristics mapping Tier -> Filename."""

  # helper to check target path
  def check_tier(tier, expected_file, op_override=None):
    # Use overridden op if provided to bypass heuristics
    target_op = op_override if op_override else sample_op

    injector = StandardsInjector(target_op, tier=tier)
    # Mock resolve to return a dummy root
    with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
      mock_resolve.return_value = Path("/mock/semantics")

      # Mock file ops to inspect path
      # Assign mock object to variable to access call args
      m_open = mock_open(read_data="{}")
      with patch("builtins.open", m_open):
        # Mock Path.mkdir and exists to prevent OS errors
        with patch("pathlib.Path.mkdir"):
          with patch("pathlib.Path.exists", return_value=True):
            injector.inject(dry_run=False)

      # Verify open called on correct file
      expected_path = Path(f"/mock/semantics/{expected_file}")

      # The first call to open should be the read (since exists=True)
      # Depending on implementation, it might open for read then write.
      # We check that expected_path was opened at least once.
      args_list = [c.args[0] for c in m_open.call_args_list]
      assert expected_path in args_list

  # Default LogSoftmax -> Neural (PascalCase heuristic)
  check_tier(SemanticTier.NEURAL, "k_neural_net.json")

  # Array tier takes precedence if explicitly passed
  check_tier(SemanticTier.ARRAY_API, "k_array_api.json")

  # EXTRAS tier gets overridden if name looks Neural (PascalCase)
  # To test EXTRAS routing, we use a utility-like name 'load_data' and explicit Extras tier
  extra_op = OperationDef(operation="manual_utility", description="util", std_args=[], variants={})
  check_tier(SemanticTier.EXTRAS, "k_framework_extras.json", op_override=extra_op)


def test_injector_appends_new_op(sample_op):
  """Verify new operation is added to the dictionary."""
  injector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)

  # Original content
  original_json = '{"Conv2d": {"description": "..."}}'

  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")

    m_open = mock_open(read_data=original_json)
    with patch("builtins.open", m_open):
      with patch("pathlib.Path.mkdir"):
        with patch("pathlib.Path.exists", return_value=True):
          injector.inject()

    # Verify write
    # We need to find the call to write().
    # m_open() returns file handle. handle.write(...) is the call.
    handle = m_open()
    written_data = "".join(str(call.args[0]) for call in handle.write.call_args_list)

    # Verify valid JSON structure was written
    data = json.loads(written_data)

    # Should have both
    assert "Conv2d" in data
    assert "LogSoftmax" in data
    assert data["LogSoftmax"]["description"] == "Log Softmax."
    assert data["LogSoftmax"]["std_args"] == ["x", "dim"]


def test_injector_updates_existing_op(sample_op):
  """Verify existing operation is updated (Overwrite behavior)."""
  injector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)

  # Original has old description
  original_json = json.dumps({"LogSoftmax": {"description": "Old Desc", "std_args": []}})

  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")

    m_open = mock_open(read_data=original_json)
    with patch("builtins.open", m_open):
      with patch("pathlib.Path.mkdir"):
        with patch("pathlib.Path.exists", return_value=True):
          injector.inject()

    handle = m_open()
    # Robustly join sequential writes
    written_data = "".join(str(call.args[0]) for call in handle.write.call_args_list)
    data = json.loads(written_data)

    # Should update description
    assert data["LogSoftmax"]["description"] == "Log Softmax."
    assert len(data["LogSoftmax"]["std_args"]) == 2


def test_injector_dry_run(sample_op, capsys):
  """Verify dry run prints to stdout instead of writing."""
  injector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)

  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")

    with patch("builtins.open", mock_open(read_data="{}")) as m_open:
      with patch("pathlib.Path.exists", return_value=True):
        injector.inject(dry_run=True)

      # Should NOT write
      handle = m_open()
      handle.write.assert_not_called()

      # Should print
      captured = capsys.readouterr()
      assert "[Dry Run]" in captured.out
      assert "LogSoftmax" in captured.out
