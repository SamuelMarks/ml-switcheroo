"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import extract_primitive_key, convert_value_to_cst


def test_extract_primitive_key_branches() -> None:
  """Docstring."""
  assert extract_primitive_key(cst.SimpleString('"test"')) == "test"
  assert extract_primitive_key(cst.Integer("1")) == "1"
  assert extract_primitive_key(cst.Name("x")) == "x"
  assert extract_primitive_key(cst.Float("1.0")) is None


def test_convert_value_to_cst_branches() -> None:
  """Docstring."""
  assert isinstance(convert_value_to_cst(True), cst.Name)
  assert isinstance(convert_value_to_cst(False), cst.Name)
  assert isinstance(convert_value_to_cst(None), cst.Name)
  assert isinstance(convert_value_to_cst(1), cst.Integer)
  assert isinstance(convert_value_to_cst(1.5), cst.Float)
  assert isinstance(convert_value_to_cst(-1.5), cst.UnaryOperation)
  assert isinstance(convert_value_to_cst("abc"), cst.SimpleString)

  lst_node: cst.CSTNode = convert_value_to_cst([1, 2])
  assert isinstance(lst_node, cst.List)
  assert len(lst_node.elements) == 2

  tup_node: cst.CSTNode = convert_value_to_cst((1,))
  assert isinstance(tup_node, cst.Tuple)

  dict_node: cst.CSTNode = convert_value_to_cst({"a": 1})
  assert isinstance(dict_node, cst.Dict)

  # Unknown
  class Unknown:
    """Docstring."""

    pass

  assert isinstance(convert_value_to_cst(Unknown()), cst.SimpleString)
