"""
Tests for Semantics Bisector.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.testing.bisector import SemanticsBisector
from ml_switcheroo.testing.runner import EquivalenceRunner


def test_propose_fix_relaxes_tolerances():
  # Mock Runner
  runner = MagicMock(spec=EquivalenceRunner)

  # 1st call fails (1e-3, 1e-4) - Standard
  # 2nd call fails (1e-3, 1e-3) - Loose Absolute
  # 3rd call passes (1e-2, 1e-3) - Loose Relative
  runner.verify.side_effect = [
    (False, "Fail"),
    (False, "Fail"),
    (True, "Pass"),
  ]

  bisector = SemanticsBisector(runner)
  op_def = {"std_args": ["x"], "variants": {"a": {}}, "test_rtol": 1e-5}

  patch = bisector.propose_fix("MyOp", op_def)

  assert patch is not None
  # Correct expectation: Step 3 is (1e-2, 1e-3). 1e-2 == 0.01.
  assert patch["test_rtol"] == 0.01
