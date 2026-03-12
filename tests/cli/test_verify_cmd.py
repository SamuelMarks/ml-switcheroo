import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from ml_switcheroo.cli.handlers.verify import handle_ci


def test_handle_ci():
  with (
    patch("ml_switcheroo.cli.handlers.verify.RuntimeConfig") as MockConfig,
    patch("ml_switcheroo.cli.handlers.verify.load_plugins", return_value=1),
    patch("ml_switcheroo.cli.handlers.verify.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as MockValidator,
    patch("pathlib.Path.exists", return_value=False),
    patch("ml_switcheroo.cli.handlers.verify.ReadmeEditor") as MockEditor,
  ):
    MockConfig.load.return_value.plugin_paths = ["fake"]
    MockValidator.return_value.run_all.return_value = {"op1": True, "op2": False}

    res = handle_ci(repair=False, update_readme=True, readme_path=Path("README.md"), json_report=None)
    assert res == 0
    MockEditor.return_value.update_matrix.assert_called_once()


def test_handle_ci_repair():
  with (
    patch("ml_switcheroo.cli.handlers.verify.RuntimeConfig.load", side_effect=Exception("error")),
    patch("ml_switcheroo.cli.handlers.verify.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as MockValidator,
    patch("ml_switcheroo.cli.handlers.verify.SemanticsBisector") as MockBisector,
    patch("builtins.open", mock_open()) as m_open,
    patch("pathlib.Path.mkdir"),
  ):
    MockValidator.return_value.run_all.return_value = {"op1": True, "op2": False, "op3": False, "op4": False}

    # Test repair success, fail, and not found
    MockSemantics.return_value.get_definition_by_id.side_effect = [None, {"def": 2}, {"def": 3}]
    MockBisector.return_value.propose_fix.side_effect = [True, False]

    res = handle_ci(repair=True, update_readme=False, readme_path=Path(""), json_report=Path("out.json"))
    assert res == 0
    m_open.assert_called_once()


def test_handle_ci_json_fail():
  with (
    patch("ml_switcheroo.cli.handlers.verify.RuntimeConfig.load", side_effect=Exception("error")),
    patch("ml_switcheroo.cli.handlers.verify.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as MockValidator,
    patch("pathlib.Path.mkdir", side_effect=Exception("mkdir error")),
  ):
    MockValidator.return_value.run_all.return_value = {}

    res = handle_ci(repair=False, update_readme=False, readme_path=Path(""), json_report=Path("out.json"))
    assert res == 1


def test_handle_ci_repair_no_fixes():
  with (
    patch("ml_switcheroo.cli.handlers.verify.RuntimeConfig.load", side_effect=Exception("error")),
    patch("ml_switcheroo.cli.handlers.verify.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as MockValidator,
    patch("ml_switcheroo.cli.handlers.verify.SemanticsBisector") as MockBisector,
  ):
    MockValidator.return_value.run_all.return_value = {"op1": False}
    MockBisector.return_value.propose_fix.return_value = False

    res = handle_ci(repair=True, update_readme=False, readme_path=Path(""), json_report=None)
    assert res == 0
