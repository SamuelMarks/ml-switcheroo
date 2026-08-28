"""Test module."""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.testing.runner import EquivalenceRunner
import numpy as np
from typing import Dict, Any


def test_runner_error_branches() -> None:
  """Test element."""
  runner: EquivalenceRunner = EquivalenceRunner()
  assert runner._deep_compare(1, 2) is False
  assert runner._deep_compare([1, 2], [1, 2]) is True
  assert runner._deep_compare([1], [1, 2]) is False
  assert runner._deep_compare(np.array([1.0]), np.array([1.00001])) is True
  assert runner._deep_compare(np.array([1.0]), np.array([2.0])) is False

  class BadIter:
    def __len__(self) -> int:
      return 1

    def __iter__(self) -> Any:
      raise Exception("bad")

  assert runner._deep_compare(BadIter(), [1]) is False

  res_dict: Dict[str, Any] = {"fw1": 1, "fw2": 1}
  runner._compare_results(res_dict, 1e-5, 1e-5, [])
  with pytest.raises(AssertionError):
    runner._compare_results({"fw1": 1, "fw2": 2}, 1e-5, 1e-5, [])

  with patch.object(runner, "_execute_api", side_effect=Exception("Mock Crash")):
    res: bool
    msg: str
    res, msg = runner.verify({"tf": {"api": "tf.add"}}, ["x"])
    assert res is False
    assert "Crash" in msg


def test_runner_hypothesis_exception() -> None:
  """Test element."""
  runner: EquivalenceRunner = EquivalenceRunner()
  # Mock fuzzer to just return a dummy strategy
  runner.fuzzer.build_strategies = MagicMock(return_value={"x": MagicMock()})

  # We want execute_api to return different things for different fws
  def mock_exec(api: str, args: Dict[str, Any]) -> int:
    if "tf" in api:
      return 1
    return 2

  runner._execute_api = mock_exec

  # Run verify with multiple fws
  res: bool
  msg: str
  res, msg = runner.verify({"tf": {"api": "tf.add"}, "torch": {"api": "torch.add"}}, ["x"])
  assert res is False
  assert "Verification Failed" in msg


def test_runner_misc_misses() -> None:
  """Test element."""
  runner: EquivalenceRunner = EquivalenceRunner()

  # 93 continue
  runner.verify({"bad_fw": "not_a_dict", "bad_fw2": {}}, [])

  # 158 return None
  assert runner._execute_api("no_dots", {}) is None

  # 228 exception in .numpy()
  class BadNumpy:
    def numpy(self) -> Any:
      raise Exception("bad numpy")

  assert runner._deep_compare(BadNumpy(), 1) is False
