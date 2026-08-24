"""Test module."""

from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.cli.commands import handle_gen_weight_script


@patch("ml_switcheroo.cli.commands.WeightScriptGenerator")
def test_handle_gen_weight_script_success(mock_gen_class):
  """Test element."""
  mock_instance = mock_gen_class.return_value
  mock_instance.generate.return_value = True

  res = handle_gen_weight_script(Path("model.py"), Path("out.py"), "torch", "jax")
  assert res == 0
  mock_instance.generate.assert_called_once_with(Path("model.py"), Path("out.py"))


@patch("ml_switcheroo.cli.commands.WeightScriptGenerator")
def test_handle_gen_weight_script_failure(mock_gen_class):
  """Test element."""
  mock_instance = mock_gen_class.return_value
  mock_instance.generate.return_value = False

  res = handle_gen_weight_script(Path("model.py"), Path("out.py"), "torch", "jax")
  assert res == 1
  mock_instance.generate.assert_called_once_with(Path("model.py"), Path("out.py"))
