"""Docstring."""

import typing

from ml_switcheroo.generated_tests.inputs import (
  _generate_array_code,
  _generate_dim_heuristic,
  _infer_type_from_default,
  generate_input_value_code,
  parse_arg_def,
)


def test_parse_arg_def() -> None:
  """Docstring."""
  # hit dict logic
  res: dict[str, typing.Any] = parse_arg_def({"name": "x"})
  assert res["type"] == "Array"

  res2: dict[str, typing.Any] = parse_arg_def({"name": "x", "default": True})
  assert res2["type"] == "bool"

  res3: dict[str, typing.Any] = parse_arg_def({"name": "x", "default": 1})
  assert res3["type"] == "int"

  res4: dict[str, typing.Any] = parse_arg_def({"name": "x", "default": 1.0})
  assert res4["type"] == "float"

  res5: dict[str, typing.Any] = parse_arg_def({"name": "x", "default": [1]})
  assert res5["type"] == "Array"


def test_infer_type() -> None:
  """Docstring."""
  assert _infer_type_from_default(True) == "bool"
  assert _infer_type_from_default(1) == "int"
  assert _infer_type_from_default(1.0) == "float"
  assert _infer_type_from_default([1, 2]) == "List[int]"
  assert _infer_type_from_default(["a", "b"]) == "List[Any]"
  assert _infer_type_from_default("foo") == "Any"


def test_generate_input() -> None:
  """Docstring."""
  # list[int]
  res1: str = generate_input_value_code("x", {"type": "List[int]", "default": [5, 6]})
  assert res1 == "[5, 6]"

  res1b: str = generate_input_value_code("x", {"type": "List[int]"})
  assert res1b == "[1, 2]"

  # tuple
  res2: str = generate_input_value_code("x", {"type": "Tuple[int]", "default": (5, 6)})
  assert res2 == "(5, 6)"

  res2b: str = generate_input_value_code("x", {"type": "Tuple[int]"})
  assert res2b == "(1, 2)"

  # int with min only
  res3: str = generate_input_value_code("x", {"type": "int", "min": 10})
  assert "randint(10, 15)" in res3

  # int with max only
  res4: str = generate_input_value_code("x", {"type": "int", "max": 10})
  assert "randint(5, 10)" in res4

  # float inference from constraint
  res5: str = generate_input_value_code("x", {"min": 1.0})
  assert "uniform" in res5

  # int inference from constraint
  res6: str = generate_input_value_code("x", {"min": 1})
  assert "randint" in res6


def test_heuristic_fallback() -> None:
  """Docstring."""
  assert "1" in _generate_dim_heuristic("dim")
  assert "bool" in _generate_dim_heuristic("keepdim")
  assert "1" in _generate_dim_heuristic("other")


def test_generate_array_code() -> None:
  """Docstring."""
  res: str = _generate_array_code({"min": 0, "max": 1, "dtype": "int32"})
  assert "astype(np.int32)" in res
  assert "uniform" in res


def test_parse_arg_def_misc() -> None:
  """Docstring."""
  assert parse_arg_def("x") == {"name": "x", "type": "Array"}
  assert parse_arg_def(("x", "int")) == {"name": "x", "type": "int"}
  assert parse_arg_def(123) == {"name": "unknown", "type": "Array"}


def test_generate_input_value_code_misc() -> None:
  """Docstring."""
  res: str = generate_input_value_code("x", {"options": [1, 2]})
  assert "random.choice" in res

  res2: str = generate_input_value_code("dim", {"type": "Any"})
  assert res2 == "1"

  res3: str = generate_input_value_code("x", {"type": "Any"})
  assert "np.random.randn" in res3

  res4: str = generate_input_value_code("x", {"type": "UnknownType"})
  assert res4 == "None"


def test_generate_array_code_misc() -> None:
  """Docstring."""
  res: str = _generate_array_code({"min": 5.0})
  assert "np.abs" in res
  assert "astype(np.float32)" in res


def test_generate_input_value_code_str_arg_def() -> None:
  """Docstring."""
  res: str = generate_input_value_code("x", "int")
  assert "randint" in res


def test_generate_input_value_code_bool() -> None:
  """Docstring."""
  res: str = generate_input_value_code("x", {"type": "bool"})
  assert "getrandbits" in res


def test_generate_input_value_code_float() -> None:
  """Docstring."""
  res: str = generate_input_value_code("x", {"type": "float", "min": 0, "max": 1})
  assert "random.uniform(0, 1)" in res

  res2: str = generate_input_value_code("x", {"type": "float"})
  assert "random.uniform" in res2


def test_generate_input_value_code_any_with_default() -> None:
  """Docstring."""
  res: str = generate_input_value_code("x", {"type": "Any", "default": True})
  assert "bool" in res
