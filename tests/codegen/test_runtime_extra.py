import pytest
import sys
import numpy as np
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.runtime import verify_results
import ml_switcheroo.generated_tests.runtime as rt


@pytest.fixture
def mock_sys_modules():
  torch_mock = MagicMock()
  tf_mock = MagicMock()

  sys.modules["torch"] = torch_mock
  sys.modules["tensorflow"] = tf_mock
  yield torch_mock, tf_mock
  del sys.modules["torch"]
  del sys.modules["tensorflow"]


def test_ensure_determinism_auto(mock_sys_modules):
  torch_mock, tf_mock = mock_sys_modules
  func = getattr(rt.ensure_determinism, "__pytest_wrapped__", None)
  if func:
    func = func.obj
  else:
    func = rt.ensure_determinism.__wrapped__

  mlx_mock = MagicMock()
  sys.modules["mlx"] = mlx_mock

  sys.modules["mlx.core"] = mlx_mock.core

  func()

  torch_mock.manual_seed.side_effect = Exception("boom")
  tf_mock.random.set_seed.side_effect = Exception("boom")
  mlx_mock.core.random.seed.side_effect = Exception("boom")
  func()

  del sys.modules["mlx.core"]
  # 64-68: mlx.core not in modules, but mlx is
  func()

  del sys.modules["mlx"]


def test_verify_results_shape_mismatch():
  # 133-134
  assert verify_results(np.array([1, 2]), np.array([1, 2, 3])) is False


def test_verify_results_types():
  assert verify_results(None, None) is True
  assert verify_results(None, 1) is False

  assert verify_results({"a": 1}, {"a": 1}) is True
  assert verify_results({"a": 1}, {"b": 1}) is False
  assert verify_results({"a": 1}, {"a": 2}) is False

  assert verify_results([1, 2], [1, 2]) is True
  assert verify_results([1, 2], [1]) is False
  assert verify_results([1, 2], [1, 3]) is False

  # numpy conversions
  assert verify_results(np.array([1.0]), np.array([1.0]), exact=True) is True
  assert verify_results(np.array([1.0]), np.array([1.0]), exact=False) is True
  assert verify_results(np.array([1]), np.array([1]), exact=False) is True

  class Uncomparable:
    def __eq__(self, other):
      raise ValueError("bad eq")

  assert verify_results(Uncomparable(), Uncomparable()) is False

  class Comparable:
    def __eq__(self, other):
      return True

  assert verify_results(Comparable(), Comparable()) is True

  class NoArray:
    def __array__(self):
      raise ValueError("no array")

    def __eq__(self, other):
      return True

  assert verify_results(NoArray(), NoArray()) is True


def test_verify_results_chex(monkeypatch):
  chex_mock = MagicMock()
  chex_mock.assert_trees_all_close = MagicMock()
  rt.globals = lambda: {"chex": chex_mock}

  assert verify_results(1, 1, exact=True) is True
  assert verify_results(1, 1, exact=False) is True
  chex_mock.assert_trees_all_close.side_effect = AssertionError("mismatch")
  assert verify_results(1, 1) is True

  # test globals()["chex"] raise exception
  chex_mock.assert_trees_all_close.side_effect = Exception("err")
  assert verify_results(1, 1) is True

  del rt.globals
