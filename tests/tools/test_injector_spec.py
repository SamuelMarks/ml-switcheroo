"""Test suite for the Injector Spec module."""

import yaml
from unittest.mock import patch, mock_open
from pathlib import Path
import pytest
from ml_switcheroo.core.dsl import OperationDef, OpType
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.tools.injector_spec import StandardsInjector


@pytest.fixture
def sample_op():
  """Provides a mock sample op for testing."""
  return OperationDef(
    operation="LogSoftmax", description="Log Softmax.", op_type=OpType.FUNCTION, std_args=["x", "dim"], variants={}
  )


def test_injector_finds_correct_file(sample_op):
  """Verifies the behavior of injector finds correct file."""

  def check_tier(tier, expected_file, op_override=None):
    """Checks tier."""
    target_op = op_override if op_override else sample_op
    injector = StandardsInjector(target_op, tier=tier)
    with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
      mock_resolve.return_value = Path("/mock/semantics")
      m_open = mock_open(read_data="{}")
      with patch("builtins.open", m_open):
        with patch("pathlib.Path.mkdir"):
          with patch("pathlib.Path.exists", return_value=True):
            injector.inject(dry_run=False)
      expected_path = Path(f"/mock/semantics/{expected_file}")
      args_list = [c.args[0] for c in m_open.call_args_list]
      assert expected_path in args_list

  check_tier(SemanticTier.NEURAL, "odl/LogSoftmax.yaml")
  extra_op2 = OperationDef(operation="abs", description="util", std_args=[], variants={})
  check_tier(SemanticTier.ARRAY_API, "odl/abs.yaml", op_override=extra_op2)
  extra_op = OperationDef(operation="manual_utility", description="util", std_args=[], variants={})
  check_tier(SemanticTier.EXTRAS, "odl/manual_utility.yaml", op_override=extra_op)


def test_injector_appends_new_op(sample_op):
  """Verifies the behavior of injector appends new op."""
  injector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)
  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")
    m_open = mock_open()
    with patch("builtins.open", m_open):
      with patch("pathlib.Path.mkdir"):
        with patch("pathlib.Path.exists", return_value=False):
          injector.inject()
    handle = m_open()
    written_data = "".join((str(call.args[0]) for call in handle.write.call_args_list))
    data = yaml.safe_load(written_data)
    assert data["operation"] == "LogSoftmax"
    assert data["description"] == "Log Softmax."
    assert data["std_args"] == ["x", "dim"]


def test_injector_updates_existing_op(sample_op):
  """Verifies the behavior of injector updates existing op."""
  injector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)
  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")
    m_open = mock_open()
    with patch("builtins.open", m_open):
      with patch("pathlib.Path.mkdir"):
        with patch("pathlib.Path.exists", return_value=True):
          injector.inject()
    handle = m_open()
    written_data = "".join((str(call.args[0]) for call in handle.write.call_args_list))
    data = yaml.safe_load(written_data)
    assert data["operation"] == "LogSoftmax"
    assert data["description"] == "Log Softmax."
    assert len(data["std_args"]) == 2


def test_injector_dry_run(sample_op, capsys):
  """Verifies the behavior of injector dry run."""
  injector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)
  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")
    with patch("builtins.open", mock_open(read_data="{}")) as m_open:
      with patch("pathlib.Path.exists", return_value=True):
        injector.inject(dry_run=True)
      handle = m_open()
      handle.write.assert_not_called()
      captured = capsys.readouterr()
      assert "[Dry Run]" in captured.out
      assert "LogSoftmax" in captured.out
