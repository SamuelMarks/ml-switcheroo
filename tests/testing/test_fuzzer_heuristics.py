"""Test suite for the Fuzzer Heuristics module."""


def test_guess_dtype_by_name():
  """Verifies the behavior of guess dtype by name."""
  from ml_switcheroo.testing.fuzzer.heuristics import guess_dtype_by_name

  assert guess_dtype_by_name("mask") == "bool"
  assert guess_dtype_by_name("is_valid") == "bool"
  assert guess_dtype_by_name("index") == "int"
  assert guess_dtype_by_name("n_items") == "int"
  assert guess_dtype_by_name("alpha") == "float"


def test_generate_by_heuristic():
  """Generates by heuristic."""
  from ml_switcheroo.testing.fuzzer.heuristics import generate_by_heuristic
  import numpy as np

  assert generate_by_heuristic("foo", (2, 2), {"options": [42]}) == 42
  assert generate_by_heuristic("axis", (2, 2, 2)) in (0, 1, 2)
  assert generate_by_heuristic("dim", (2,)) == 0
  assert generate_by_heuristic("dim", ()) == 0
  assert generate_by_heuristic("keepdims", (2,)) in (True, False)
  assert generate_by_heuristic("shape", (2, 3)) == (2, 3)
  arr = generate_by_heuristic("foo", (2, 2), {"dtype": "int32"})
  assert arr.dtype == np.int32
  arr = generate_by_heuristic("foo", (2, 2), {"dtype": "bool"})
  assert arr.dtype == np.bool_
  arr = generate_by_heuristic("mask", (2, 2))
  assert arr.dtype == np.bool_
  assert isinstance(generate_by_heuristic("val_index", ()), int)
  arr = generate_by_heuristic("indices", (2, 2))
  assert arr.dtype == np.int32
  assert isinstance(generate_by_heuristic("alpha", ()), float)
  arr = generate_by_heuristic("inputs", (2, 2))
  assert arr.dtype == np.float32
