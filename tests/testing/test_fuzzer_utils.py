def test_is_pipe_top_level():
  from ml_switcheroo.testing.fuzzer.utils import is_pipe_top_level

  assert is_pipe_top_level("int | str")
  assert not is_pipe_top_level("List[int | str]")
  assert not is_pipe_top_level("Tuple[int, float]")
  assert is_pipe_top_level("List[int] | Dict[str, Any]")


def test_split_outside_brackets():
  from ml_switcheroo.testing.fuzzer.utils import split_outside_brackets

  assert split_outside_brackets("int, str") == ["int", "str"]
  assert split_outside_brackets("List[int, str], float") == ["List[int, str]", "float"]
  assert split_outside_brackets("Tuple[int], Dict[str, Any]") == ["Tuple[int]", "Dict[str, Any]"]


def test_resolve_symbolic_shape():
  from ml_switcheroo.testing.fuzzer.utils import resolve_symbolic_shape

  sym_map = {}
  shape = resolve_symbolic_shape("'B', 32, N", sym_map)
  assert len(shape) == 3
  assert shape[1] == 32
  assert "B" in sym_map
  assert "N" in sym_map

  # Check reuse
  shape2 = resolve_symbolic_shape("B, N", sym_map)
  assert shape2[0] == shape[0]
  assert shape2[1] == shape[2]

  # complex non-ident
  shape3 = resolve_symbolic_shape("N+1", sym_map)
  assert len(shape3) == 1

  # empty
  assert resolve_symbolic_shape("'', \"\"", sym_map) == ()


def test_adjust_shape_rank():
  from ml_switcheroo.testing.fuzzer.utils import adjust_shape_rank

  assert adjust_shape_rank((2, 2), 2) == (2, 2)

  padded = adjust_shape_rank((2, 2), 4)
  assert len(padded) == 4
  assert padded[:2] == (2, 2)

  truncated = adjust_shape_rank((2, 2, 2, 2), 2)
  assert truncated == (2, 2)
