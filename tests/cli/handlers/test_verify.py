"""Test suite for the Verify module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.handlers.verify import handle_ci


@pytest.fixture
def mock_config():
  """Provides a mock configuration for testing."""
  with patch("ml_switcheroo.config.RuntimeConfig.load") as mock_load:
    mock_conf = MagicMock()
    mock_conf.plugin_paths = ["some/plugin"]
    mock_load.return_value = mock_conf
    yield mock_load


@pytest.fixture
def mock_semantics():
  """Provides a mock semantics for testing."""
  with patch("ml_switcheroo.cli.handlers.verify.SemanticsManager") as mock:
    yield mock


@pytest.fixture
def mock_validator():
  """Provides a mock validator for testing."""
  with patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as mock:
    yield mock


@pytest.fixture
def mock_load_plugins():
  """Provides a mock load plugins for testing."""
  with patch("ml_switcheroo.cli.handlers.verify.load_plugins") as mock:
    yield mock


def test_handle_ci_success(mock_config, mock_semantics, mock_validator, mock_load_plugins):
  """Handles ci successfully."""
  mock_load_plugins.return_value = 1
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {"Op1": True, "Op2": False}
  assert handle_ci(False, Path("README.md"), None, False) == 0


def test_handle_ci_plugin_load_none(mock_config, mock_semantics, mock_validator, mock_load_plugins):
  """Handles ci when loaded plugins is 0."""
  mock_load_plugins.return_value = 0
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {}
  assert handle_ci(False, Path("README.md"), None, False) == 0


def test_handle_ci_repair_no_defn(mock_config, mock_semantics, mock_validator):
  """Handles ci repair when no definition is found."""
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {"Op1": False}
  mock_sem_instance = mock_semantics.return_value
  mock_sem_instance.get_definition_by_id.return_value = None
  assert handle_ci(False, Path("README.md"), None, True) == 0


def test_handle_ci_config_error(mock_config, mock_semantics, mock_validator):
  """Handles ci configuration correctly handling an error."""
  mock_config.side_effect = Exception("boom")
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {}
  assert handle_ci(False, Path("README.md"), None, False) == 0


def test_handle_ci_repair(mock_config, mock_semantics, mock_validator):
  """Handles ci repair."""
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {"Op1": True, "Op2": False, "Op3": False}
  mock_sem_instance = mock_semantics.return_value
  mock_sem_instance.get_definition_by_id.side_effect = [{"variants": {}}, None]
  with patch("ml_switcheroo.cli.handlers.verify.SemanticsBisector") as mock_bisector:
    mock_bis_inst = mock_bisector.return_value
    mock_bis_inst.propose_fix.side_effect = [{"test_rtol": 0.1}, None]
    assert handle_ci(False, Path("README.md"), None, True) == 0
    assert mock_sem_instance.update_definition.call_count == 1


def test_handle_ci_repair_no_fixes(mock_config, mock_semantics, mock_validator):
  """Handles ci repair no fixes."""
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {"Op1": True, "Op2": False}
  mock_sem_instance = mock_semantics.return_value
  mock_sem_instance.get_definition_by_id.return_value = {"some": "definition"}
  with patch("ml_switcheroo.cli.handlers.verify.SemanticsBisector") as mock_bisector:
    mock_bis_inst = mock_bisector.return_value
    mock_bis_inst.propose_fix.return_value = None
    assert handle_ci(False, Path("README.md"), None, True) == 0


def test_handle_ci_update_readme(mock_config, mock_semantics, mock_validator):
  """Handles ci update readme."""
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {}
  with patch("ml_switcheroo.cli.handlers.verify.ReadmeEditor") as mock_editor:
    assert handle_ci(True, Path("README.md"), None, False) == 0
    mock_editor.return_value.update_matrix.assert_called_once()


def test_handle_ci_json_report(mock_config, mock_semantics, mock_validator, tmp_path):
  """Handles ci JSON report."""
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {"Op1": True}
  report_path = tmp_path / "report.json"
  assert handle_ci(False, Path("README.md"), report_path, False) == 0
  assert report_path.exists()


def test_handle_ci_json_report_error(mock_config, mock_semantics, mock_validator, tmp_path):
  """Handles ci JSON report correctly handling an error."""
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {"Op1": True}
  report_path = tmp_path / "report.json"
  with patch("pathlib.Path.mkdir", side_effect=Exception("boom")):
    assert handle_ci(False, Path("README.md"), report_path, False) == 1


def test_handle_ci_no_tests_dir(mock_config, mock_semantics, mock_validator):
  """Handles ci no tests directory."""
  mock_val_instance = mock_validator.return_value
  mock_val_instance.run_all.return_value = {}
  with patch("pathlib.Path.exists", return_value=False):
    assert handle_ci(False, Path("README.md"), None, False) == 0
