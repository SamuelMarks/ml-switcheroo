"""Test module."""

import json
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.handlers.verify import handle_ci


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator")
@patch("ml_switcheroo.cli.handlers.verify.ReadmeEditor")
def test_handle_ci_success(mock_readme_class, mock_validator_class, tmp_path):
  """Test element."""
  mock_validator = mock_validator_class.return_value
  mock_validator.run_all.return_value = {"Op1": True, "Op2": False}

  mock_readme = mock_readme_class.return_value

  report_file = tmp_path / "report.json"
  readme_file = tmp_path / "README.md"

  res = handle_ci(update_readme=True, readme_path=readme_file, json_report=report_file, repair=False)

  assert res == 0
  mock_readme.update_matrix.assert_called_once_with({"Op1": True, "Op2": False})

  assert report_file.exists()
  data = json.loads(report_file.read_text())
  assert data == {"Op1": True, "Op2": False}


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator")
@patch("ml_switcheroo.cli.handlers.verify.SemanticsBisector")
@patch("ml_switcheroo.cli.handlers.verify.SemanticsManager")
def test_handle_ci_repair(mock_sm_class, mock_bisect_class, mock_val_class, tmp_path):
  """Test element."""
  mock_sm = mock_sm_class.return_value
  mock_sm.get_definition_by_id.side_effect = lambda op: {"def": op} if op == "Op2" else None

  mock_val = mock_val_class.return_value
  mock_val.run_all.return_value = {"Op1": True, "Op2": False, "Op3": False}

  mock_bisect = mock_bisect_class.return_value
  mock_bisect.propose_fix.side_effect = lambda op, d: {"patch": True} if op == "Op2" else None

  res = handle_ci(False, tmp_path / "README.md", None, repair=True)

  assert res == 0
  mock_sm.update_definition.assert_called_once_with("Op2", {"patch": True})


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator")
def test_handle_ci_config_load_error(mock_val, tmp_path):
  """Test element."""
  with patch("ml_switcheroo.cli.handlers.verify.RuntimeConfig.load", side_effect=Exception("Boom")):
    # Should not crash, just logs warning
    res = handle_ci(False, tmp_path / "README.md", None, False)
    assert res == 0


@patch("ml_switcheroo.cli.handlers.verify.RuntimeConfig.load")
@patch("ml_switcheroo.cli.handlers.verify.load_plugins")
@patch("ml_switcheroo.cli.handlers.verify.BatchValidator")
def test_handle_ci_load_plugins(mock_val, mock_load_plugins, mock_load_config, tmp_path):
  """Test element."""
  mock_config = MagicMock()
  mock_config.plugin_paths = ["some/path"]
  mock_load_config.return_value = mock_config
  mock_load_plugins.return_value = 1

  res = handle_ci(False, tmp_path / "README.md", None, False)
  assert res == 0
  mock_load_plugins.assert_called_once_with(extra_dirs=["some/path"])


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator")
def test_handle_ci_report_fail(mock_val, tmp_path):
  """Test element."""
  mock_val.return_value.run_all.return_value = {}

  # Pass a directory as json_report to force an exception
  bad_path = tmp_path / "bad_dir"
  bad_path.mkdir()

  res = handle_ci(False, tmp_path / "README.md", bad_path, False)
  assert res == 1


def test_handle_ci_manual_test_dir_exists(tmp_path):
  """Test element."""
  with (
    patch("ml_switcheroo.cli.handlers.verify.Path.exists", return_value=True),
    patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as mock_val,
  ):
    handle_ci(False, tmp_path / "README.md", None, False)
    # Verify run_all was called with manual_test_dir=Path('tests')
    # We can't strictly assert the exact Path object easily if it uses relative 'tests'
    # but we can check it wasn't None.
    call_kwargs = mock_val.return_value.run_all.call_args[1]
    assert call_kwargs["manual_test_dir"] is not None


def test_handle_ci_manual_test_dir_not_exists(tmp_path):
  """Test element."""
  # If the real tests dir exists, we need to mock it returning False
  with (
    patch("ml_switcheroo.cli.handlers.verify.Path.exists", return_value=False),
    patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as mock_val,
  ):
    handle_ci(False, tmp_path / "README.md", None, False)
    call_kwargs = mock_val.return_value.run_all.call_args[1]
    assert call_kwargs["manual_test_dir"] is None


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator")
@patch("ml_switcheroo.cli.handlers.verify.SemanticsBisector")
@patch("ml_switcheroo.cli.handlers.verify.SemanticsManager")
def test_handle_ci_repair_no_fixes(mock_sm_class, mock_bisect_class, mock_val_class, tmp_path):
  """Test element."""
  mock_sm = mock_sm_class.return_value
  mock_sm.get_definition_by_id.return_value = {"def": "op"}

  mock_val = mock_val_class.return_value
  mock_val.run_all.return_value = {"Op1": False}  # Fail

  mock_bisect = mock_bisect_class.return_value
  mock_bisect.propose_fix.return_value = None  # No patch proposed

  res = handle_ci(False, tmp_path / "README.md", None, repair=True)
  assert res == 0
