"""Test suite for commands."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.commands import handle_gen_weight_script


@patch("ml_switcheroo.cli.commands.WeightScriptGenerator")
def test_handle_gen_weight_script_success(mock_gen_class: MagicMock) -> None:
  """Verify handle gen weight script success behavior.

  Args:
      mock_gen_class: Mock object for gen class.
  """
  mock_instance: MagicMock = mock_gen_class.return_value
  mock_instance.generate.return_value = True

  res: int = handle_gen_weight_script(Path("model.py"), Path("out.py"), "torch", "jax")
  assert res == 0
  mock_instance.generate.assert_called_once_with(Path("model.py"), Path("out.py"))


@patch("ml_switcheroo.cli.commands.WeightScriptGenerator")
def test_handle_gen_weight_script_failure(mock_gen_class: MagicMock) -> None:
  """Verify handle gen weight script failure behavior.

  Args:
      mock_gen_class: Mock object for gen class.
  """
  mock_instance: MagicMock = mock_gen_class.return_value
  mock_instance.generate.return_value = False

  res: int = handle_gen_weight_script(Path("model.py"), Path("out.py"), "torch", "jax")
  assert res == 1
  mock_instance.generate.assert_called_once_with(Path("model.py"), Path("out.py"))


@mock.patch("ml_switcheroo.cli.commands.RuntimeConfig")
@mock.patch("ml_switcheroo.cli.commands.SemanticsManager")
@mock.patch("ml_switcheroo.cli.commands.WeightScriptGenerator")
def test_handle_gen_weight_script(
  mock_generator_cls: mock.MagicMock,
  mock_sem_mgr: mock.MagicMock,
  mock_rc: mock.MagicMock,
) -> None:
  """Verify handle gen weight script behavior.

  Args:
      mock_generator_cls: Mock object for generator cls.
      mock_sem_mgr: Mock object for sem mgr.
      mock_rc: Mock object for rc.
  """
  # Setup mocks
  mock_config: mock.Mock = mock.Mock()
  mock_rc.load.return_value = mock_config

  mock_sem: mock.Mock = mock.Mock()
  mock_sem_mgr.return_value = mock_sem

  mock_generator: mock.Mock = mock.Mock()
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
