"""Unit tests for `EquivalenceRunner` and `_evaluate_shape_calc` (`ml_switcheroo.testing.runner`)."""

import sys
import typing
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import numpy as np
import pytest
from pydantic import JsonValue

from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.testing.runner import EquivalenceRunner, _evaluate_shape_calc


@pytest.fixture
def mock_frameworks(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
  """Register lightweight `torch` and `jax.numpy` stubs returning `np.array(5.0)` in `sys.modules`.

  Args:
      monkeypatch: Pytest monkeypatch fixture.

  Returns:
      Dictionary of registered framework module mocks.
  """

  def _make_mod(name: str) -> MagicMock:
    """Create a non-iterable `MagicMock` module stub.

    Args:
        name: Module name string.

    Returns:
        Configured `MagicMock`.
    """
    m = MagicMock(name=name)
    m.__iter__.side_effect = TypeError(f"'{name}' object is not iterable")
    return m

  mock_torch = _make_mod("torch")
  mock_torch.sum.return_value = np.array(5.0)
  mock_jax = _make_mod("jax")
  mock_jnp = _make_mod("jax.numpy")
  mock_jax.numpy = mock_jnp
  mock_jnp.sum.return_value = np.array(5.0)

  overrides = {"torch": mock_torch, "jax": mock_jax, "jax.numpy": mock_jnp}
  for mod_name, mod_obj in overrides.items():
    monkeypatch.setitem(sys.modules, mod_name, mod_obj)
  return overrides


def test_equivalence_flow_and_adapter_normalization(mock_frameworks: dict[str, MagicMock]) -> None:
  """Verify `EquivalenceRunner.verify` across matching variants, shape checks, crashes, and `NumpyAdapter`.

  Args:
      mock_frameworks: Stubbed `torch` and `jax` module dictionary.
  """
  runner = EquivalenceRunner()
  ok, msg = runner.verify({"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}, params=["x"])
  assert ok is True
  assert "Verified" in msg

  mock_adapter = MagicMock()
  mock_adapter.convert.return_value = np.array(5.0)
  with patch("ml_switcheroo.testing.runner.get_adapter", return_value=mock_adapter) as mock_get:
    runner.verify({"torch": {"api": "torch.sum"}}, params=["x"])
    mock_get.assert_called_with("numpy")
    mock_adapter.convert.assert_called()

  # Shape calculations: exact match, mismatch, syntax error, runtime ZeroDivisionError, and `get_adapter=None`
  assert runner.verify({"torch": {"api": "torch.sum"}}, params=["x"], shape_calc="lambda x: ()")[0] is True
  ok_mis, msg_mis = runner.verify({"torch": {"api": "torch.sum"}}, params=["x"], shape_calc="lambda x: (1, 2)")
  assert ok_mis is False
  assert "Shape Mismatch" in msg_mis

  ok_syn, msg_syn = runner.verify({"torch": {"api": "torch.sum"}}, params=["x"], shape_calc="lambda x:")
  assert ok_syn is False
  assert "Shape Calculation Error" in msg_syn

  mock_frameworks["torch"].bad_api.side_effect = RuntimeError("Mock Crash")
  ok_crash, msg_crash = runner.verify({"torch": {"api": "torch.bad_api"}}, params=["x"])
  assert ok_crash is False
  assert "Crash in torch" in msg_crash

  class _DetachableTensorStub:
    """Typed tensor stub implementing `detach().cpu().numpy()`."""

    def detach(self) -> "_DetachableTensorStub":
      """Return self.

      Returns:
          `_DetachableTensorStub` instance.
      """
      return self

    def cpu(self) -> "_DetachableTensorStub":
      """Return self.

      Returns:
          `_DetachableTensorStub` instance.
      """
      return self

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
      """Convert to scalar NumPy array `1.0`.

      Returns:
          Scalar float64 NumPy array.
      """
      return np.array(1.0)

  np_adapter = NumpyAdapter()
  stub = _DetachableTensorStub()
  arr = np.array([1.0, 2.0])
  assert np_adapter.convert(stub) == np.array(1.0)
  conv_arr = np_adapter.convert(arr)
  assert isinstance(conv_arr, np.ndarray)
  assert np.allclose(conv_arr, arr)
  converted_list = np_adapter.convert([stub, arr])
  assert isinstance(converted_list, list)
  assert converted_list[0] == np.array(1.0)
  converted_dict = np_adapter.convert({"k": stub})
  assert isinstance(converted_dict, dict)
  assert converted_dict["k"] == np.array(1.0)


def test_runner_execute_compare_and_deep_compare_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `_execute_api`, `_compare_results`, `_deep_compare`, and Hypothesis mismatch reporting.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  runner = EquivalenceRunner()
  assert runner._execute_api("no_dots", {}) is None
  with pytest.raises(TypeError, match="not callable"):
    runner._execute_api("math.pi", {})

  # Non-dict / missing-api variants and optional-import skips
  assert runner.verify({"bad1": "not_a_dict", "bad2": {}}, [])[0] is True
  assert runner.verify({"missing_fw": {"api": "missing_fw_pkg_xyz.op"}}, [])[0] is True

  with patch.object(runner, "_execute_api", side_effect=ImportError("optional missing")):
    assert runner.verify({"torch": {"api": "torch.sum"}, "jax": {"api": "jax.sum"}}, ["x"])[0] is True

  with patch.object(runner, "_execute_api", return_value=42):
    assert runner.verify({"torch": {"api": "torch.sum"}}, ["x"], shape_calc="lambda x: (1,)")[0] is True
    assert runner.verify({"torch": {"api": "torch.sum"}}, ["y"], shape_calc="lambda y: (1,)")[0] is True

  class _FixedFuzzer:
    """Deterministic fuzzer stub yielding `x = np.array([1])`."""

    def build_strategies(
      self,
      _p: list[str],
      _h: dict[str, str] | None = None,
      _c: dict[str, dict[str, JsonValue]] | None = None,
    ) -> dict[str, st.SearchStrategy[np.ndarray[tuple[int, ...], np.dtype[np.int64]]]]:
      """Build constant search strategy for `x`.

      Args:
          _p: Parameter names.
          _h: Parameter type hints.
          _c: Parameter constraints.

      Returns:
          Strategy mapping for `x`.
      """
      return {"x": st.just(np.array([1]))}

    def adapt_to_framework(
      self,
      args: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.int64]]],
      _fw: str,
    ) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.int64]]]:
      """Return `args` unchanged.

      Args:
          args: Generated inputs.
          _fw: Target framework name.

      Returns:
          Unchanged `args` dictionary.
      """
      return args

  monkeypatch.setattr(runner, "fuzzer", _FixedFuzzer())
  with patch.object(runner, "_execute_api", side_effect=lambda api, _a: 1 if "tf" in api else 2):
    ok_hyp, msg_hyp = runner.verify({"tf": {"api": "tf.add"}, "torch": {"api": "torch.add"}}, ["x"])
    assert ok_hyp is False
    assert "Verification Failed" in msg_hyp

  with (
    patch.object(runner, "_execute_api", return_value=np.array([1])),
    patch("ml_switcheroo.testing.runner.get_adapter", return_value=None),
  ):
    assert runner.verify({"jax": {"api": "jax.add"}}, ["x"], {"x": "Array"}, {}, shape_calc="lambda x: (1,)")[0] is True

  # `_compare_results` and `_deep_compare` across scalars, sequences, object arrays, and broken objects
  err_box: list[str] = []
  runner._compare_results({"fw1": 1, "fw2": 1}, 1e-5, 1e-5, err_box)
  with pytest.raises(AssertionError):
    runner._compare_results({"fw1": 1, "fw2": 2}, 1e-5, 1e-5, err_box)
  assert len(err_box) == 1

  assert runner._deep_compare("s", "s") is True
  assert runner._deep_compare(1, 2) is False
  assert runner._deep_compare([1, 2], [1, 2]) is True
  assert runner._deep_compare([1], [1, 2]) is False
  assert runner._deep_compare(np.array([1.0]), np.array([1.000001])) is True
  assert runner._deep_compare(np.array([1.0]), np.array([2.0])) is False
  assert runner._deep_compare(np.array([1, 2]), np.array([1, 2, 3])) is False
  assert runner._deep_compare(np.array(["a", "b"]), np.array(["a", "b"])) is True
  assert runner._deep_compare(np.array(["a", "b"]), np.array(["a", "c"])) is False

  class _DistinctObj:
    """Distinct object instance for object-dtype array inequality."""

  assert runner._deep_compare(np.array([_DistinctObj()]), np.array([_DistinctObj()])) is False

  class _BadIter:
    """Iterable raising `RuntimeError` during iteration."""

    def __len__(self) -> int:
      """Return length `1`.

      Returns:
          Constant `1`.
      """
      return 1

    def __iter__(self) -> typing.Iterator[int]:
      """Raise `RuntimeError`.

      Returns:
          Never returns; raises `RuntimeError`.

      Raises:
          RuntimeError: Always raised.
      """
      raise RuntimeError("bad iter")

  class _BadNumpyMethod:
    """Object whose `.numpy()` raises `RuntimeError`."""

    def numpy(self) -> typing.Never:
      """Raise `RuntimeError`.

      Returns:
          Never returns; raises `RuntimeError`.

      Raises:
          RuntimeError: Always raised.
      """
      raise RuntimeError("bad numpy")

  class _BadArrayProtocol:
    """Object whose `__array__` raises `ValueError`."""

    def __array__(self, *args: JsonValue, **kwargs: JsonValue) -> typing.Never:
      """Raise `ValueError` on NumPy array coercion.

      Args:
          *args: Positional arguments.
          **kwargs: Keyword arguments.

      Returns:
          Never returns; raises `ValueError`.

      Raises:
          ValueError: Always raised.
      """
      _ = (args, kwargs)
      raise ValueError("cannot convert")

  assert runner._deep_compare(_BadIter(), [1]) is False
  assert runner._deep_compare(_BadNumpyMethod(), 1) is False
  assert runner._deep_compare(np.array([1]), _BadArrayProtocol()) is False


def test_evaluate_shape_calc_supported_and_rejected_ast_nodes() -> None:
  """Verify `_evaluate_shape_calc` supports safe arithmetic/slice ASTs and rejects unsafe constructs."""
  arr = np.zeros((4, 8))
  assert _evaluate_shape_calc("lambda x: x.shape", arr) == (4, 8)
  assert _evaluate_shape_calc("lambda x: [x.shape[0] + 1, x.shape[1] - 2]", arr) == [5, 6]
  assert _evaluate_shape_calc("lambda x: (x.shape[0] * 2, x.shape[1] // 2, x.shape[1] / 2, x.shape[1] % 3)", arr) == (
    8,
    4,
    4.0,
    2,
  )
  assert _evaluate_shape_calc("lambda x: tuple(x.shape[:1])", arr) == (4,)
  assert _evaluate_shape_calc("lambda x: list(x.shape)", arr) == [4, 8]
  assert _evaluate_shape_calc("lambda x: (int(x.shape[0]), max(x.shape[0], 6), min(x.shape[0], 2.5))", arr) == (
    4,
    6,
    2.5,
  )
  assert _evaluate_shape_calc("lambda x: (-1, len(x.shape))", arr) == (-1, 2)

  for bad_expr, match_str in (
    ("x.shape", "Expected single-argument lambda"),
    ("lambda x: unknown_var", "Unknown identifier"),
    ("lambda x: x.__class__", "Private attribute"),
    ("lambda x: x.ndim", "Unsupported attribute"),
    ("lambda x: x[0]", "must be a shape sequence"),
    ("lambda x: x.shape['a':]", "Slice bounds"),
    ("lambda x: x.shape['a']", "Subscript index"),
    ("lambda x: -x.shape", "requires numeric operand"),
    ("lambda x: max('bad')", "Unsupported call"),
    ("lambda x: eval('1')", "Unsupported call"),
    ("lambda x: (lambda y: y)(1)", "Unsupported AST node"),
    ("lambda x: {1: 2}", "Unsupported AST node"),
    ("lambda x: x.shape[0] ** 2", "Unsupported AST node"),
  ):
    with pytest.raises(ValueError, match=match_str):
      _evaluate_shape_calc(bad_expr, arr)
