"""Docstring."""

from unittest.mock import MagicMock
from ml_switcheroo.testing.batch_runner import BatchValidator


def test_batch_runner_init():
  """Docstring."""
  runner = BatchValidator(semantics=MagicMock())
  assert runner.semantics is not None
