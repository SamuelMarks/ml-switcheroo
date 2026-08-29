"""Test suite for the Runtime Comparator module coverage."""

import sys
import typing
from unittest import mock

import numpy as np
import pytest

# We need to import the fixture but avoid pytest seeing it as a test fixture directly
# if we just want to call its code. We can extract the original function.
from ml_switcheroo.generated_tests import runtime
from ml_switcheroo.generated_tests.runtime import verify_results


def test_verify_results_shape_mismatch_size_1() -> None:
  """Verifies the behavior of shape mismatch size 1."""
  assert verify_results(np.array([1.0]), 1.0)


def test_verify_results_chex(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of chex module."""
  chex_mod: mock.MagicMock = mock.MagicMock()
  # Mock doesn't allow methods starting with assert by default
  chex_mod.assert_trees_all_close = mock.MagicMock()
  monkeypatch.setitem(globals(), "chex", chex_mod)
  monkeypatch.setitem(sys.modules, "chex", chex_mod)
  # But verify_results looks for 'chex' in globals() of its own module
  monkeypatch.setattr(runtime, "chex", chex_mod, raising=False)

  assert verify_results(1, 1, exact=True)
  chex_mod.assert_trees_all_close.assert_called_with(1, 1, rtol=0, atol=0)

  assert verify_results(1, 1, exact=False)
  chex_mod.assert_trees_all_close.assert_called_with(1, 1, rtol=1e-3, atol=1e-4)

  chex_mod.assert_trees_all_close.side_effect = AssertionError("Mocked")
  # Fallback to manual recursive comparison
  assert verify_results(1, 1)


def test_verify_results_fallback() -> None:
  """Verifies the fallback exception block."""

  class BadThing:
    def __eq__(self, other: typing.Any) -> bool:
      raise ValueError("Bad")

  class VeryBadThing:
    def __init__(self, val: typing.Any) -> None:
      self.val = val

    def __array__(self) -> np.ndarray:
      raise RuntimeError("No array")

    def __eq__(self, other: typing.Any) -> bool:
      raise RuntimeError("No eq")

  assert not verify_results(VeryBadThing(1), VeryBadThing(2))


# To test ensure_determinism without fixture errors, we can just get the unwrapped func
# Or parse it out. In pytest 8, we can use __pytest_wrapped__.obj or __wrapped__ if available.
def test_ensure_determinism() -> None:
  """Verifies ensure determinism."""
  func: typing.Callable[[], None] = getattr(runtime.ensure_determinism, "__wrapped__", runtime.ensure_determinism)
  if hasattr(runtime.ensure_determinism, "__pytest_wrapped__"):
    func = getattr(runtime.ensure_determinism, "__pytest_wrapped__").obj

  with mock.patch.dict(
    "sys.modules",
    {"torch": mock.MagicMock(), "tensorflow": mock.MagicMock(), "mlx.core": mock.MagicMock(), "mlx": mock.MagicMock()},
  ):
    sys.modules["torch"].cuda.is_available.return_value = True  # type: ignore
    func()

    # Try the exceptions
    sys.modules["torch"].manual_seed.side_effect = Exception("err")  # type: ignore
    sys.modules["tensorflow"].random.set_seed.side_effect = Exception("err")  # type: ignore
    sys.modules["mlx.core"].random.seed.side_effect = Exception("err")  # type: ignore
    sys.modules["mlx"].core.random.seed.side_effect = Exception("err")  # type: ignore
    func()

  with mock.patch.dict("sys.modules", {"mlx": mock.MagicMock()}):
    sys.modules["mlx"].core = mock.MagicMock()  # type: ignore
    func()
    sys.modules["mlx"].core.random.seed.side_effect = Exception("err")  # type: ignore
    func()
