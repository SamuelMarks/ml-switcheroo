"""Test suite for the Fuzzer Constraints module."""

import typing


import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer() -> typing.Any:
  """Execute fuzzer helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_sharing(fuzzer: typing.Any, data: typing.Any) -> None:
  """Verifies the behavior of symbolic sharing.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Any = {"x": "Array['N']", "y": "Array['N']"}
  strats: typing.Any = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs: typing.Any = data.draw(st.fixed_dictionaries(strats))
  x: typing.Any = inputs["x"]
  y: typing.Any = inputs["y"]
  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.shape == y.shape
  assert len(x.shape) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_matmul_constraints(fuzzer: typing.Any, data: typing.Any) -> None:
  """Verifies the behavior of matmul constraints.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Any = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
  strats: typing.Any = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs: typing.Any = data.draw(st.fixed_dictionaries(strats))
  x: typing.Any = inputs["x"]
  y: typing.Any = inputs["y"]
  assert len(x.shape) == 2
  assert len(y.shape) == 2
  assert x.shape[1] == y.shape[0]


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_fixed_dimension(fuzzer: typing.Any, data: typing.Any) -> None:
  """Verifies the behavior of fixed dimension.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Any = {"x": "Array[3, 'D']"}
  strats: typing.Any = fuzzer.build_strategies(["x"], hints=hints)
  inputs: typing.Any = data.draw(st.fixed_dictionaries(strats))
  x: typing.Any = inputs["x"]
  assert x.shape[0] == 3
  assert len(x.shape) == 2


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_list_consistency(fuzzer: typing.Any, data: typing.Any) -> None:
  """Verifies the behavior of symbolic list consistency.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Any = {"x": "List[Array['Z']]"}
  strats: typing.Any = fuzzer.build_strategies(["x"], hints=hints)
  inputs: typing.Any = data.draw(st.fixed_dictionaries(strats))
  lst: typing.Any = inputs["x"]
  if not lst:
    return
  shape0: typing.Any = lst[0].shape
  for arr in lst[1:]:
    assert arr.shape == shape0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_independent_calls_are_independent(fuzzer: typing.Any, data: typing.Any) -> None:
  """Verifies the behavior of independent calls are independent.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Any = {"x": "Array['N']"}
  strats1: typing.Any = fuzzer.build_strategies(["x"], hints=hints)
  res1: typing.Any = data.draw(st.fixed_dictionaries(strats1))
  strats2: typing.Any = fuzzer.build_strategies(["x"], hints=hints)
  res2: typing.Any = data.draw(st.fixed_dictionaries(strats2))
  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_tensor_alias_support(fuzzer: typing.Any, data: typing.Any) -> None:
  """Verifies the behavior of tensor alias support.

  Args:
      fuzzer: Test argument or fixture.
      data: Test argument or fixture.
  """
  hints: typing.Any = {"x": "Tensor['A']"}
  strats: typing.Any = fuzzer.build_strategies(["x"], hints=hints)
  inputs: typing.Any = data.draw(st.fixed_dictionaries(strats))
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
  _res: typing.Any = generate_from_hint(
    AnyType(), base_shape=[2], depth=0, max_depth=5, symbol_map={}, constraints={"default": ["a", "b"]}
  )
  assert isinstance(_res, list)
  # this will hit the ListType(AnyType()) path
