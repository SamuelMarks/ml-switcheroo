"""Docstring."""

from unittest import mock
from pathlib import Path
from ml_switcheroo.cli.commands import handle_gen_weight_script


@mock.patch("ml_switcheroo.cli.commands.RuntimeConfig")
@mock.patch("ml_switcheroo.cli.commands.SemanticsManager")
@mock.patch("ml_switcheroo.cli.commands.WeightScriptGenerator")
def test_handle_gen_weight_script(mock_generator_cls, mock_sem_mgr, mock_rc):
  """Docstring."""
  # Setup mocks
  mock_config = mock.Mock()
  mock_rc.load.return_value = mock_config

  mock_sem = mock.Mock()
  mock_sem_mgr.return_value = mock_sem

  mock_generator = mock.Mock()
  mock_generator_cls.return_value = mock_generator

  # Test success
  mock_generator.generate.return_value = True
  assert handle_gen_weight_script(Path("in.py"), Path("out.py"), "torch", "jax") == 0
  mock_rc.load.assert_called_with(source="torch", target="jax")
  mock_generator_cls.assert_called_with(mock_sem, mock_config)
  mock_generator.generate.assert_called_with(Path("in.py"), Path("out.py"))

  # Test failure
  mock_generator.generate.return_value = False
  assert handle_gen_weight_script(Path("in.py"), Path("out.py")) == 1
