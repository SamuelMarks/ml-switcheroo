"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import extract_primitive_key, convert_value_to_cst


def test_extract_primitive_key():
  """Function docstring."""
  assert extract_primitive_key(cst.SimpleString("'hello'")) == "hello"
  assert extract_primitive_key(cst.SimpleString('"world"')) == "world"
  assert extract_primitive_key(cst.Integer("42")) == "42"
  assert extract_primitive_key(cst.Name("my_var")) == "my_var"
  # Unsupported type
  assert extract_primitive_key(cst.Float("3.14")) is None


def test_convert_value_to_cst_primitives():
  """Function docstring."""
  # bool
  node = convert_value_to_cst(True)
  assert isinstance(node, cst.Name) and node.value == "True"
  node = convert_value_to_cst(False)
  assert isinstance(node, cst.Name) and node.value == "False"

  # int
  node = convert_value_to_cst(42)
  assert isinstance(node, cst.Integer) and node.value == "42"

  # float
  node = convert_value_to_cst(3.14)
  assert isinstance(node, cst.Float) and node.value in ("3.14", "3.1400000000000001")  # repr(3.14)

  # str
  node = convert_value_to_cst("hello")
  assert isinstance(node, cst.SimpleString) and node.value == '"hello"'

  # None
  node = convert_value_to_cst(None)
  assert isinstance(node, cst.Name) and node.value == "None"

  # Unknown (fallback)
  class Dummy:
    """Class docstring."""

    def __str__(self):
      """Function docstring."""
      return "dummy"

  node = convert_value_to_cst(Dummy())
  assert isinstance(node, cst.SimpleString) and node.value == "'dummy'"


def test_convert_value_to_cst_containers():
  """Function docstring."""
  # list
  node = convert_value_to_cst([1, 2])
  assert isinstance(node, cst.List)
  assert len(node.elements) == 2
  assert isinstance(node.elements[0].value, cst.Integer)
  assert node.elements[0].comma is not cst.MaybeSentinel.DEFAULT
  assert node.elements[-1].comma is cst.MaybeSentinel.DEFAULT

  # tuple
  node = convert_value_to_cst((3, 4))
  assert isinstance(node, cst.Tuple)
  assert len(node.elements) == 2

  # dict
  node = convert_value_to_cst({"a": 1, "b": 2})
  assert isinstance(node, cst.Dict)
  assert len(node.elements) == 2
  assert isinstance(node.elements[0].key, cst.SimpleString)
  assert isinstance(node.elements[0].value, cst.Integer)
  assert node.elements[-1].comma is cst.MaybeSentinel.DEFAULT
