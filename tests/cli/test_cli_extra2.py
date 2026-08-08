"""Test suite for CLI extra 2 coverage."""

import pytest
from unittest.mock import patch, mock_open
from ml_switcheroo.cli.__main__ import main


def test_main_dispatch():
  """Test main dispatch."""
  # Test all branches in main.py

  # 1. Missing args (should exit)
  with patch("sys.argv", ["ml-switcheroo"]):
    with pytest.raises(SystemExit):
      main()

  # 2. Version
  with pytest.raises(SystemExit) as e:
    main(["--version"])
  assert e.value.code == 0

  # 3. Help
  with pytest.raises(SystemExit) as e:
    main(["--help"])
    assert e.value.code == 0

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


def test_main_dispatch_default_sys_argv():
  """Test main dispatch default sys argv."""
  with patch("sys.argv", ["ml-switcheroo", "matrix"]):
    with patch("ml_switcheroo.cli.__main__.commands.handle_matrix", return_value=0):
      assert main() == 0
