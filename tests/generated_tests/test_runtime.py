"""Canonical unit tests for ``ml_switcheroo.generated_tests.runtime``."""

import dataclasses
import sys
import types
from unittest import mock

import numpy as np
import pytest

from ml_switcheroo.generated_tests import runtime
from ml_switcheroo.generated_tests.runtime import (
  ensure_determinism as ensure_determinism,
)
from ml_switcheroo.generated_tests.runtime import (
  seed_framework_rngs,
  verify_results,
)


def test_verify_results_shape_mismatch_size_1() -> None:
  """Verify scalar vs size-1 1D NumPy array equivalence in ``verify_results``."""
  with mock.patch("importlib.util.find_spec", return_value=None):
    assert verify_results(np.array([1.0]), 1.0)
    assert verify_results(np.array([1]), np.array([1]), exact=True)


def test_verify_results_chex(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify ``chex.assert_trees_all_close`` integration, ``AssertionError``, and ``TypeError`` fallback.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """

  class ChexMod(types.ModuleType):
    """Typed ``chex`` module stub exposing ``assert_trees_all_close``."""

    assert_trees_all_close: mock.MagicMock

  chex_mod = ChexMod("chex")
  close_mock = mock.MagicMock()
  chex_mod.assert_trees_all_close = close_mock
  monkeypatch.setitem(sys.modules, "chex", chex_mod)

  assert verify_results(1, 1, exact=True)
  close_mock.assert_called_with(1, 1, rtol=0, atol=0)

  assert verify_results(1, 1, exact=False)
  close_mock.assert_called_with(1, 1, rtol=1e-3, atol=1e-4)

  close_mock.side_effect = AssertionError("Mismatch")
  assert not verify_results(1, 2)

  close_mock.side_effect = TypeError("Unsupported non-PyTree leaf")
  assert verify_results(1, 1)


def test_verify_results_fallback_and_containers() -> None:
  """Verify ``verify_results`` across dicts, lists, None, primitives, and non-array leaves."""

  @dataclasses.dataclass
  class NonArrayLeaf:
    """Custom non-array value whose ``__array__`` hook raises ``RuntimeError``."""

    val: int

    def __array__(self) -> None:
      """Reject NumPy array coercion."""
      raise RuntimeError("No array")

  with mock.patch("importlib.util.find_spec", return_value=None):
    assert verify_results(NonArrayLeaf(1), NonArrayLeaf(1))
    assert not verify_results(NonArrayLeaf(1), NonArrayLeaf(2))

    assert verify_results(None, None)
    assert not verify_results(None, 1)
    assert not verify_results(1, None)

    assert verify_results({"a": 1}, {"a": 1})
    assert not verify_results({"a": 1}, {"b": 1})
    assert not verify_results({"a": 1}, {"a": 2})

    assert verify_results([1, 2], [1, 2])
    assert not verify_results([1, 2], [1, 3])
    assert not verify_results([1, 2], [1])

    assert not verify_results(np.array([1, 2]), np.array([1, 2, 3]))

    assert verify_results("hello", "hello")
    assert not verify_results("hello", "world")


def test_ensure_determinism_and_seed_framework_rngs() -> None:
  """Verify ``ensure_determinism`` and ``seed_framework_rngs`` across all framework protocols."""
  assert runtime.ensure_determinism is not None

  class TorchMod(types.ModuleType):
    """Typed Torch stub module with ``manual_seed`` and ``cuda``."""

    manual_seed: mock.MagicMock
    cuda: types.SimpleNamespace

  class TfMod(types.ModuleType):
    """Typed TensorFlow stub module with ``random``."""

    random: types.SimpleNamespace

  class MlxCoreMod(types.ModuleType):
    """Typed MLX core stub module with ``random``."""

    random: types.SimpleNamespace

  class MlxRootMod(types.ModuleType):
    """Typed MLX root stub module with ``core``."""

    core: MlxCoreMod

  class KerasMod(types.ModuleType):
    """Typed Keras stub module with ``utils``."""

    utils: types.SimpleNamespace

  torch_mod = TorchMod("torch")
  torch_mod.manual_seed = mock.MagicMock()
  cuda_avail = mock.MagicMock(return_value=True)
  cuda_seed_all = mock.MagicMock()
  torch_mod.cuda = types.SimpleNamespace(is_available=cuda_avail, manual_seed_all=cuda_seed_all)

  tf_mod = TfMod("tensorflow")
  tf_set_seed = mock.MagicMock()
  tf_mod.random = types.SimpleNamespace(set_seed=tf_set_seed)

  mlx_core = MlxCoreMod("mlx.core")
  mlx_seed = mock.MagicMock()
  mlx_core.random = types.SimpleNamespace(seed=mlx_seed)

  keras_mod = KerasMod("keras")
  keras_seed = mock.MagicMock()
  keras_mod.utils = types.SimpleNamespace(set_random_seed=keras_seed)

  with mock.patch.dict(
    sys.modules,
    {
      "torch": torch_mod,
      "tensorflow": tf_mod,
      "mlx.core": mlx_core,
      "keras": keras_mod,
    },
    clear=False,
  ):
    seed_framework_rngs(42)
    torch_mod.manual_seed.assert_called_with(42)
    cuda_seed_all.assert_called_with(42)
    tf_set_seed.assert_called_with(42)
    mlx_seed.assert_called_with(42)
    keras_seed.assert_called_with(42)

    # Test CUDA unavailable branch first before raising RuntimeError
    cuda_avail.return_value = False
    seed_framework_rngs(7)

    # Test RuntimeError resilience
    torch_mod.manual_seed.side_effect = RuntimeError("torch err")
    tf_set_seed.side_effect = RuntimeError("tf err")
    mlx_seed.side_effect = RuntimeError("mlx err")
    keras_seed.side_effect = RuntimeError("keras err")
    seed_framework_rngs(7)

    # Test non-protocol cuda/random/utils objects and missing cuda attribute
    torch_mod.manual_seed.side_effect = None
    torch_mod.cuda = types.SimpleNamespace()
    tf_mod.random = types.SimpleNamespace(seed=mock.MagicMock())
    mlx_core.random = types.SimpleNamespace(set_seed=mock.MagicMock())
    keras_mod.utils = types.SimpleNamespace()
    seed_framework_rngs(7)
    torch_no_cuda = types.SimpleNamespace(manual_seed=mock.MagicMock())
    with mock.patch.dict(sys.modules, {"torch": torch_no_cuda}, clear=False):
      seed_framework_rngs(8)

  # Test root 'mlx' module fallback when 'mlx.core' is absent from sys.modules
  mlx_root = MlxRootMod("mlx")
  mlx_core2 = MlxCoreMod("mlx.core")
  mlx_seed2 = mock.MagicMock()
  mlx_core2.random = types.SimpleNamespace(seed=mlx_seed2)
  mlx_root.core = mlx_core2
  saved_mlx_core = sys.modules.pop("mlx.core", None)
  try:
    with mock.patch.dict(sys.modules, {"mlx": mlx_root}, clear=False):
      seed_framework_rngs(11)
      mlx_seed2.assert_called_once_with(11)
  finally:
    if saved_mlx_core is not None:
      sys.modules["mlx.core"] = saved_mlx_core

  with mock.patch("ml_switcheroo.frameworks.base.get_adapter", return_value=None):
    seed_framework_rngs(1)
