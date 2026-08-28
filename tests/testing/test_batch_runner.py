"""Docstring."""

from unittest.mock import MagicMock
from ml_switcheroo.testing.batch_runner import BatchValidator


def test_batch_runner_init() -> None:
  """Docstring."""
  runner: BatchValidator = BatchValidator(semantics=MagicMock())
  assert getattr(runner, "semantics") is not None
