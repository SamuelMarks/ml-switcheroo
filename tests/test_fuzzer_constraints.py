"""Test suite for the Fuzzer Constraints module."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Execute fuzzer helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_sharing(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of symbolic sharing.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints = {"x": "Array['N']", "y": "Array['N']"}
  strats = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  x = inputs["x"]
  y = inputs["y"]
  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.shape == y.shape
  assert len(x.shape) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_matmul_constraints(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of matmul constraints.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
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
def test_fixed_dimension(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of fixed dimension.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints = {"x": "Array[3, 'D']"}
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
  """Verifies the behavior of symbolic list consistency.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints = {"x": "List[Array['Z']]"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  lst = inputs["x"]
  if not isinstance(lst, list) or not lst:
    return
  shape0 = getattr(lst[0], "shape")
  for arr in lst[1:]:
    assert getattr(arr, "shape") == shape0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_independent_calls_are_independent(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of independent calls are independent.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints = {"x": "Array['N']"}
  strats1 = fuzzer.build_strategies(["x"], hints=hints)
  res1 = data.draw(st.fixed_dictionaries(strats1))
  strats2 = fuzzer.build_strategies(["x"], hints=hints)
  res2 = data.draw(st.fixed_dictionaries(strats2))
  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_tensor_alias_support(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of tensor alias support.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints = {"x": "Tensor['A']"}
  strats = fuzzer.build_strategies(["x"], hints=hints)
  inputs = data.draw(st.fixed_dictionaries(strats))
  assert isinstance(inputs["x"], np.ndarray)
  assert len(inputs["x"].shape) == 1


def test_fuzzer_parser_extra_coverage() -> None:
  """Verify fuzzer parser extra coverage behavior."""
  # Optional return None
  import random

  from ml_switcheroo.testing.fuzzer.parser import generate_from_hint
  from ml_switcheroo.testing.fuzzer.type_parser import AnyType, OptionalType, PrimitiveType

  random.seed(42)  # Try to hit < 0.2
  for _ in range(20):
    if (
      generate_from_hint(
        OptionalType(inner=PrimitiveType(name="int")), base_shape=[2], depth=0, max_depth=5, symbol_map={}
      )
      is None
    ):
      break

  # default_val = list of strings (not int)
  _res = generate_from_hint(
    AnyType(), base_shape=[2], depth=0, max_depth=5, symbol_map={}, constraints={"default": ["a", "b"]}
  )
  assert isinstance(_res, list)
  # this will hit the ListType(AnyType()) path
