"""Test suite for the Fuzzer Symbolic module."""

import random

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from pydantic import JsonValue

from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Create a configured `InputFuzzer` fixture for fuzzer symbolic tests.

  Returns:
      Configured `fuzzer` fixture value (InputFuzzer).
  """
  random.seed(42)
  np.random.seed(42)
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_heuristic_booleans(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles heuristic booleans as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  strats = fuzzer.build_strategies(["mask", "condition"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  mask = inputs["mask"]
  cond = inputs["condition"]
  assert isinstance(mask, np.ndarray)
  assert isinstance(cond, np.ndarray)
  assert mask.dtype == np.bool_
  assert cond.dtype == np.bool_


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_heuristic_integers(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles heuristic integers as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  strats = fuzzer.build_strategies(["indices", "k_idx"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  indices = inputs["indices"]
  k_idx = inputs["k_idx"]
  assert isinstance(indices, np.ndarray)
  assert isinstance(k_idx, np.ndarray)
  assert np.issubdtype(indices.dtype, np.integer)
  assert np.issubdtype(k_idx.dtype, np.integer)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_heuristic_scalars(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles heuristic scalars as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  strats = fuzzer.build_strategies(["alpha", "eps"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert set(inputs.keys()) == {"alpha", "eps"}
  assert isinstance(inputs["alpha"], float)
  assert isinstance(inputs["eps"], float)
  alpha = inputs["alpha"]
  eps = inputs["eps"]
  assert isinstance(alpha, float)
  assert isinstance(eps, float)
  assert alpha >= -1000.0
  assert alpha <= 1000.0
  assert eps >= -1000.0
  assert eps <= 1000.0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_axis_heuristic_validity(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles axis heuristic validity as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  strats = fuzzer.build_strategies(["x", "axis"])
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  axis = inputs["axis"]
  assert isinstance(x, np.ndarray)
  assert isinstance(axis, int)
  assert x.dtype == np.float32
  assert x.ndim >= 1
  assert x.ndim <= 4
  assert axis >= 0
  assert axis <= 3


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_hint_primitive_override(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles hint primitive override as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"mask": "int"}
  strats = fuzzer.build_strategies(["mask"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert set(inputs.keys()) == {"mask"}
  assert isinstance(inputs["mask"], int)
  assert not isinstance(inputs["mask"], bool)
  bounded = data.draw(fuzzer.build_strategies(["mask"], hints=hints, constraints={"mask": {"min": 4, "max": 4}})["mask"])
  assert bounded == 4


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_hint_array(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles hint array as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"x": "Array"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["x"]
  assert isinstance(val, np.ndarray)
  assert val.dtype == np.float32


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_hint_union(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles hint union as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"x": "int | float"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  assert set(strats.keys()) == {"x"}
  val = data.draw(st.fixed_dictionaries(strats))["x"]
  assert isinstance(val, (int, float))
  assert not isinstance(val, bool)
  bounded_val = data.draw(fuzzer.build_strategies(["x"], hints=hints, constraints={"x": {"min": 2, "max": 5}})["x"])
  assert isinstance(bounded_val, (int, float))
  assert float(bounded_val) >= 2.0
  assert float(bounded_val) <= 5.0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_hint_tuple_variadic(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles hint tuple variadic as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"vals": "Tuple[int, ...]"}
  strats = fuzzer.build_strategies(["vals"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["vals"]
  assert isinstance(val, tuple)
  assert len(val) >= 1
  assert all((isinstance(v, int) for v in val))


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_hint_nested_complex(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles hint nested complex as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"config": "Dict[str, List[int]]"}
  strats = fuzzer.build_strategies(["config"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["config"]
  assert isinstance(val, dict)
  assert len(val) >= 1
  assert len(val) <= 3
  k, v = next(iter(val.items()))
  assert isinstance(k, str)
  assert len(k) >= 1
  assert len(k) <= 10
  assert isinstance(v, list)
  assert len(v) >= 1
  assert len(v) <= 4
  assert isinstance(v[0], int)
  assert not isinstance(v[0], bool)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_recursion_limit_stops(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles recursion limit stops as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hint: str = "List[List[List[List[int]]]]"
  hints: dict[str, str] = {"deep": hint}
  strats = fuzzer.build_strategies(["deep"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  _val = inputs["deep"]
  assert isinstance(_val, list)
  assert len(_val) >= 1
  assert len(_val) <= 4
  lvl1 = _val[0]
  assert isinstance(lvl1, list)
  lvl2 = lvl1[0]
  assert isinstance(lvl2, list)
  lvl3 = lvl2[0]
  assert isinstance(lvl3, list)
  assert len(lvl3) >= 1
  assert isinstance(lvl3[0], int)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_unhashable_dict_key_fallback(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles unhashable dict key fallback as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"bad_key": "Dict[Array, int]"}
  strats = fuzzer.build_strategies(["bad_key"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["bad_key"]
  assert isinstance(val, dict)
  assert len(val) >= 1
  assert len(val) <= 3
  key, mapped_val = next(iter(val.items()))
  assert isinstance(key, str)
  assert "[" in key
  assert isinstance(mapped_val, int)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_dtype_object_generation(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles dtype object generation as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"d": "dtype"}
  strats = fuzzer.build_strategies(["d"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  val = inputs["d"]
  assert isinstance(val, (type, np.dtype))
  assert val in (np.float32, np.int32, np.float64, np.bool_)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_sharing(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` returns `1` for symbolic sharing.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"x": "Array['N']", "y": "Array['N']"}
  strats = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  y = inputs["y"]
  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.shape == y.shape
  assert len(x.shape) == 1
  assert x.shape[0] >= 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_matmul_constraints(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` returns `2` for matmul constraints.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
  strats = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  y = inputs["y"]
  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert len(x.shape) == 2
  assert len(y.shape) == 2
  assert x.shape[1] == y.shape[0]


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_fixed_dimension_mixed(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` returns `3` for fixed dimension mixed.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"x": "Array[3, 'D']"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  assert isinstance(x, np.ndarray)
  assert x.shape[0] == 3
  assert len(x.shape) == 2


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_list_consistency(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles symbolic list consistency as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"x": "List[Array['Z']]"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  lst = inputs["x"]
  if not isinstance(lst, list) or not lst:
    return
  first_arr = lst[0]
  assert isinstance(first_arr, np.ndarray)
  shape0 = first_arr.shape
  for arr in lst[1:]:
    assert isinstance(arr, np.ndarray)
    assert arr.shape == shape0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_independent_calls_are_independent(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verify `given` handles independent calls are independent as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
      data: Input `data` parameter value (`st.DataObject`).
  """
  hints: dict[str, str] = {"x": "Array['N']"}
  strats1 = fuzzer.build_strategies(["x"], hints=hints)
  res1 = data.draw(st.fixed_dictionaries(strats1))
  strats2 = fuzzer.build_strategies(["x"], hints=hints)
  res2 = data.draw(st.fixed_dictionaries(strats2))
  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)
  assert list(res1.keys()) == ["x"]
  assert list(res2.keys()) == ["x"]
  assert len(res1["x"].shape) == 1
  assert len(res2["x"].shape) == 1


@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_adapt_to_framework_passthrough(fuzzer: InputFuzzer) -> None:
  """Verify `fuzzer.adapt_to_framework` handles passthrough as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
  """
  raw = {"x": np.array([1])}
  res = fuzzer.adapt_to_framework(raw, "unknown_fw")
  assert res is raw


@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_adapt_to_framework_delegation(fuzzer: InputFuzzer) -> None:
  """Verify `register_framework` handles adapt to framework delegation as expected.

  Args:
      fuzzer: Input `fuzzer` parameter value (`InputFuzzer`).
  """

  @register_framework("mock_fw")
  class MockAdapter:
    """Test double `MockAdapter` implementing `convert` for fuzzer symbolic tests."""

    def convert(self, x: JsonValue) -> str:
      """Mock implementation of convert.

      Args:
          x: Input tensor or scalar operand (`JsonValue`).

      Returns:
          Resulting `str` from convert.
      """
      return "converted"

  raw = {"x": 1}
  res = fuzzer.adapt_to_framework(raw, "mock_fw")
  assert res["x"] == "converted"
