"""Module docstring."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_cov_287():
  """Function docstring."""
  parser = MlirParser("^bb0: ^bb1:")
  blk = parser.parse_block()
  assert blk.label == "^bb0"
  assert len(blk.operations) == 0
