def test_generate_scalar_int():
  from ml_switcheroo.testing.fuzzer.generators import generate_scalar_int

  assert isinstance(generate_scalar_int({}), int)
  assert generate_scalar_int({"min": 10, "max": 10}) == 10


def test_generate_scalar_float():
  from ml_switcheroo.testing.fuzzer.generators import generate_scalar_float

  res = generate_scalar_float({"min": 10, "max": 20})
  assert 10 <= res <= 20

  res = generate_scalar_float({"min": 10})
  assert res >= 10

  res = generate_scalar_float({"max": -10})
  assert res <= -10

  res = generate_scalar_float({})
  assert isinstance(res, float)


def test_generate_array():
  from ml_switcheroo.testing.fuzzer.generators import generate_array
  import numpy as np

  # float
  arr = generate_array("float", (2, 2), {})
  assert arr.shape == (2, 2)
  assert arr.dtype == np.float32

  # int
  arr = generate_array("int", (2, 2), {})
  assert arr.dtype == np.int32

  # bool
  arr = generate_array("bool", (2, 2), {})
  assert arr.dtype == np.bool_

  # explicit dtype bool
  arr = generate_array("float", (2, 2), {"dtype": "bool"})
  assert arr.dtype == np.bool_

  # explicit dtype int
  arr = generate_array("float", (2, 2), {"dtype": "int16"})
  assert arr.dtype == np.int16

  # explicit dtype int with bounds
  arr = generate_array("int", (2, 2), {"min": 5, "max": 10, "dtype": "int16"})
  assert arr.dtype == np.int16
  assert (arr >= 5).all() and (arr <= 10).all()

  # explicit dtype float
  arr = generate_array("int", (2, 2), {"dtype": "float64"})
  assert arr.dtype == np.float64

  # bounds for float
  arr = generate_array("float", (2, 2), {"min": 5, "max": 10})
  assert (arr >= 5).all() and (arr <= 10).all()

  arr = generate_array("float", (2, 2), {"min": 5})
  assert (arr >= 5).all()

  arr = generate_array("float", (2, 2), {"max": -5})
  assert (arr <= -5).all()

  # invalid dtype
  arr = generate_array("float", (2, 2), {"dtype": "invalid_type"})
  assert arr.dtype == np.float32


def test_get_random_shape():
  from ml_switcheroo.testing.fuzzer.generators import get_random_shape

  shape = get_random_shape((3, 3))
  assert shape == (3, 3)

  shape = get_random_shape()
  assert 1 <= len(shape) <= 4


def test_make_broadcastable_shape():
  from ml_switcheroo.testing.fuzzer.generators import make_broadcastable_shape

  shape = make_broadcastable_shape((10, 10, 10))
  assert len(shape) == 3
  assert all(d in (1, 10) for d in shape)


def test_generate_fake_callable():
  from ml_switcheroo.testing.fuzzer.generators import generate_fake_callable

  fn = generate_fake_callable()
  assert fn(42) == 42
