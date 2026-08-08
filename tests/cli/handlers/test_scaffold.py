"""Test suite for the Scaffold module."""

from argparse import Namespace
from unittest.mock import patch, mock_open
from ml_switcheroo.cli.handlers.scaffold import handle_scaffold


def test_handle_scaffold(capsys):
  """Handles scaffold.

  Args:
      capsys: ...
  """
  args = Namespace(framework="jax.numpy")
  with (
    patch("ml_switcheroo.cli.handlers.scaffold.ConsensusEngine") as MockEngine,
    patch("builtins.open", mock_open()) as mock_file,
  ):
    mock_instance = MockEngine.return_value
    mock_instance.cluster.return_value = {"add": ["jax.numpy.add"], "sub": []}
    handle_scaffold(args)
    mock_file.assert_called_with("jax.numpy_skeleton.json", "w")
    captured = capsys.readouterr()
    assert "Scaffolding API mapping for framework: jax.numpy" in captured.out
    assert "Skeleton written to jax.numpy_skeleton.json (Found 2 candidate ops)" in captured.out
