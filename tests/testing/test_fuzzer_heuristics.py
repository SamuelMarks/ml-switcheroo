def test_guess_dtype_by_name():
  from ml_switcheroo.testing.fuzzer.heuristics import guess_dtype_by_name

  assert guess_dtype_by_name("mask") == "bool"
  assert guess_dtype_by_name("is_valid") == "bool"
  assert guess_dtype_by_name("index") == "int"
  assert guess_dtype_by_name("n_items") == "int"
  assert guess_dtype_by_name("alpha") == "float"


def test_generate_by_heuristic():
  from ml_switcheroo.testing.fuzzer.heuristics import generate_by_heuristic
  import numpy as np

  # options
  assert generate_by_heuristic("foo", (2, 2), {"options": [42]}) == 42

  # axis
  assert generate_by_heuristic("axis", (2, 2, 2)) in (0, 1, 2)
  assert generate_by_heuristic("dim", (2,)) == 0
  assert generate_by_heuristic("dim", ()) == 0

  # keepdims
  assert generate_by_heuristic("keepdims", (2,)) in (True, False)

  # shape
  assert generate_by_heuristic("shape", (2, 3)) == (2, 3)

  # dtype requests
  arr = generate_by_heuristic("foo", (2, 2), {"dtype": "int32"})
  assert arr.dtype == np.int32
  arr = generate_by_heuristic("foo", (2, 2), {"dtype": "bool"})
  assert arr.dtype == np.bool_

  # bool heuristic
  arr = generate_by_heuristic("mask", (2, 2))
  assert arr.dtype == np.bool_

  # int scalar heuristic
  assert isinstance(generate_by_heuristic("val_index", ()), int)

  # int array heuristic
  arr = generate_by_heuristic("indices", (2, 2))
  assert arr.dtype == np.int32

  # float scalar heuristic
  assert isinstance(generate_by_heuristic("alpha", ()), float)

  # float array heuristic
  arr = generate_by_heuristic("inputs", (2, 2))
  assert arr.dtype == np.float32
