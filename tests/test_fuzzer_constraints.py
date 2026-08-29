"""Test suite for the Fuzzer Constraints module."""

from typing import Dict, List

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ml_switcheroo.testing.fuzzer import InputFuzzer


@pytest.fixture
def fuzzer() -> InputFuzzer:
  """Docstring."""
  return InputFuzzer()


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_sharing(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of symbolic sharing."""
  hints: Dict[str, str] = {"x": "Array['N']", "y": "Array['N']"}
  strats: Dict[str, st.SearchStrategy[np.ndarray]] = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs: Dict[str, np.ndarray] = data.draw(st.fixed_dictionaries(strats))
  x: np.ndarray = inputs["x"]
  y: np.ndarray = inputs["y"]
  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.shape == y.shape
  assert len(x.shape) == 1


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_matmul_constraints(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of matmul constraints."""
  hints: Dict[str, str] = {"x": "Array['A', 'B']", "y": "Array['B', 'C']"}
  strats: Dict[str, st.SearchStrategy[np.ndarray]] = fuzzer.build_strategies(["x", "y"], hints=hints)
  inputs: Dict[str, np.ndarray] = data.draw(st.fixed_dictionaries(strats))
  x: np.ndarray = inputs["x"]
  y: np.ndarray = inputs["y"]
  assert len(x.shape) == 2
  assert len(y.shape) == 2
  assert x.shape[1] == y.shape[0]


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_fixed_dimension(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of fixed dimension."""
  hints: Dict[str, str] = {"x": "Array[3, 'D']"}
  strats: Dict[str, st.SearchStrategy[np.ndarray]] = fuzzer.build_strategies(["x"], hints=hints)
  inputs: Dict[str, np.ndarray] = data.draw(st.fixed_dictionaries(strats))
  x: np.ndarray = inputs["x"]
  assert x.shape[0] == 3
  assert len(x.shape) == 2


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_symbolic_list_consistency(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of symbolic list consistency."""
  hints: Dict[str, str] = {"x": "List[Array['Z']]"}
  strats: Dict[str, st.SearchStrategy[List[np.ndarray]]] = fuzzer.build_strategies(["x"], hints=hints)
  inputs: Dict[str, List[np.ndarray]] = data.draw(st.fixed_dictionaries(strats))
  lst: List[np.ndarray] = inputs["x"]
  if not lst:
    return
  shape0: tuple[int, ...] = lst[0].shape
  for arr in lst[1:]:
    assert arr.shape == shape0


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_independent_calls_are_independent(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of independent calls are independent."""
  hints: Dict[str, str] = {"x": "Array['N']"}
  strats1: Dict[str, st.SearchStrategy[np.ndarray]] = fuzzer.build_strategies(["x"], hints=hints)
  res1: Dict[str, np.ndarray] = data.draw(st.fixed_dictionaries(strats1))
  strats2: Dict[str, st.SearchStrategy[np.ndarray]] = fuzzer.build_strategies(["x"], hints=hints)
  res2: Dict[str, np.ndarray] = data.draw(st.fixed_dictionaries(strats2))
  assert isinstance(res1["x"], np.ndarray)
  assert isinstance(res2["x"], np.ndarray)


@given(data=st.data())
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.skip(reason="Fuzzer constraints timeout")
def test_tensor_alias_support(fuzzer: InputFuzzer, data: st.DataObject) -> None:
  """Verifies the behavior of tensor alias support."""
  hints: Dict[str, str] = {"x": "Tensor['A']"}
  strats: Dict[str, st.SearchStrategy[np.ndarray]] = fuzzer.build_strategies(["x"], hints=hints)
  inputs: Dict[str, np.ndarray] = data.draw(st.fixed_dictionaries(strats))
  assert isinstance(inputs["x"], np.ndarray)
  assert len(inputs["x"].shape) == 1


def test_fuzzer_parser_extra_coverage() -> None:
  """Docstring."""
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
  generate_from_hint(AnyType(), base_shape=[2], depth=0, max_depth=5, symbol_map={}, constraints={"default": ["a", "b"]})
  # this will hit the ListType(AnyType()) path
