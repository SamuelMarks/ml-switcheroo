"""Test suite for CLI extra coverage."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.cli.commands import handle_gen_weight_script
from ml_switcheroo.cli.handlers.dev import handle_docs, handle_gen_tests, handle_matrix


def test_gen_weight_script_failure() -> None:
  """Docstring."""
  # Coverage for failure in commands.py handle_gen_weight_script
  with patch("ml_switcheroo.cli.commands.WeightScriptGenerator.generate", return_value=False):
    assert handle_gen_weight_script(Path("in.py"), Path("out.py")) == 1


def test_dev_handlers() -> None:
  """Docstring."""
  # Cover dev handler operations
  MockMatrix: MagicMock
  with patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix") as MockMatrix:
    mock_matrix: MagicMock = MagicMock()
    MockMatrix.return_value = mock_matrix
    assert handle_matrix() == 0
    mock_matrix.render.assert_called_once()

  MockGen1: MagicMock
  with (
    patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator") as MockGen1,
    patch("builtins.open", mock_open()),
    patch("ml_switcheroo.cli.handlers.dev.SemanticsManager"),
  ):
    mock_gen_instance1: MagicMock = MagicMock()
    mock_gen_instance1.generate.return_value = "Docs"
    MockGen1.return_value = mock_gen_instance1
    assert handle_docs("torch", "jax", Path("docs.md")) == 0
    mock_gen_instance1.generate.assert_called_once()

  MockGen2: MagicMock
  with (
    patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator") as MockGen2,
    patch("ml_switcheroo.cli.handlers.dev.Path.mkdir"),
  ):
    mock_gen_instance2: MagicMock = MagicMock()
    MockGen2.return_value = mock_gen_instance2
    assert handle_gen_tests(Path("out.py")) == 0
    mock_gen_instance2.generate.assert_called_once()


# --- Merged from test_cli_extra2.py ---


def test_main_dispatch() -> None:
  """Docstring."""
  # Test all branches in main.py

  # 1. Missing args (should exit)
  with patch("sys.argv", ["ml-switcheroo"]):
    with pytest.raises(SystemExit):
      main()

  # 2. Version
  with pytest.raises(SystemExit) as e1:
    main(["--version"])
  assert e1.value.code == 0

  # 3. Help
  with pytest.raises(SystemExit) as e2:
    main(["--help"])
  assert e2.value.code == 0

  # 4. Unknown command
  with pytest.raises(SystemExit):
    main(["unknown_command"])

  # 5. Convert command
  with patch("ml_switcheroo.cli.__main__.commands.handle_convert", return_value=0):
    assert main(["convert", "in.py"]) == 0

  # 6. Verify command
  with patch("ml_switcheroo.cli.__main__.commands.handle_ci", return_value=0):
    assert main(["ci"]) == 0

  # 7. Scaffold command
  with patch("ml_switcheroo.cli.__main__.handle_scaffold", return_value=0):
    assert main(["scaffold", "MyOp"]) == 0

  # 8. Meta schema command
  with patch("ml_switcheroo.cli.__main__.handle_schema", return_value=0):
    assert main(["schema"]) == 0

  # 9. Suggest command
  with patch("ml_switcheroo.cli.__main__.handle_suggest", return_value=0):
    assert main(["suggest", "torch"]) == 0

  # 10. Harvest command
  with patch("ml_switcheroo.cli.__main__.handle_harvest", return_value=0):
    assert main(["harvest", "torch"]) == 0

  # 11. Weight-Script command
  with patch("ml_switcheroo.cli.__main__.commands.handle_gen_weight_script", return_value=0):
    assert main(["gen-weight-script", "in.py", "--out", "out.py"]) == 0

  # 12. Matrix command
  with patch("ml_switcheroo.cli.__main__.commands.handle_matrix", return_value=0):
    assert main(["matrix"]) == 0

  # 13. Gen-Docs command
  with patch("ml_switcheroo.cli.__main__.commands.handle_docs", return_value=0):
    assert main(["gen-docs", "--source", "torch", "--target", "jax", "--out", "out.md"]) == 0

  # 14. Gen-Tests command
  with patch("ml_switcheroo.cli.__main__.commands.handle_gen_tests", return_value=0):
    assert main(["gen-tests", "--out", "out.py"]) == 0

  # 15. Verified pipeline command (success)
  with (
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline", return_value={"status": "success"}),
    patch("builtins.open", mock_open(read_data="import torch")),
  ):
    assert main(["verified-pipeline", "in.py"]) == 0

  # 16. Verified pipeline command (failure)
  with (
    patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline", return_value={"status": "failed"}),
    patch("builtins.open", mock_open(read_data="import torch")),
  ):
    assert main(["verified-pipeline", "in.py"]) == 1


def test_main_dispatch_default_sys_argv() -> None:
  """Docstring."""
  with patch("sys.argv", ["ml-switcheroo", "matrix"]):
    with patch("ml_switcheroo.cli.__main__.commands.handle_matrix", return_value=0):
      assert main() == 0
