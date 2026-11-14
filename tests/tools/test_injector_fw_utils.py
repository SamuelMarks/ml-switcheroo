"""Test suite for the Injector Fw Utils Missing module."""


def test_injector_fw_utils_missing() -> None:
  """Verify `get_import_root` handles injector fw utils missing as expected."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import (
    convert_to_cst_literal,
    get_import_root,
    is_docstring,
    is_future_import,
  )

  assert get_import_root(cst.Integer("1")) == ""
  assert is_docstring(cst.SimpleStatementLine([]), 0) is False
  assert is_docstring(cst.SimpleStatementLine([cst.Pass()]), 0) is False
  assert is_future_import(cst.SimpleStatementLine([cst.Pass()])) is False
  assert (
    is_future_import(
      cst.SimpleStatementLine([cst.ImportFrom(module=cst.Name("math"), names=[cst.ImportAlias(name=cst.Name("pi"))])])
    )
    is False
  )
  assert is_future_import(cst.Integer("1")) is False
  from unittest import mock

  with mock.patch("libcst.parse_expression", side_effect=Exception("fail")):
    assert isinstance(convert_to_cst_literal("something"), cst.SimpleString)

  class UnrecognizedObject:
    """Sentinel `UnrecognizedObject` fixture class used in injector fw utils tests."""

  assert isinstance(convert_to_cst_literal(UnrecognizedObject()), cst.SimpleString)


def test_injector_fw_utils_convert_complex() -> None:
  """Verify `convert_to_cst_literal` handles injector fw utils convert complex as expected."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal

  assert isinstance(convert_to_cst_literal([1, 2]), cst.List)
  assert isinstance(convert_to_cst_literal((1, 2)), cst.Tuple)
  assert isinstance(convert_to_cst_literal({"a": 1, "b": 2}), cst.Dict)
  none_node = convert_to_cst_literal(None)
  assert isinstance(none_node, cst.Name)
  assert none_node.value == "None"
  true_node = convert_to_cst_literal(True)
  assert isinstance(true_node, cst.Name)
  assert true_node.value == "True"
  assert isinstance(convert_to_cst_literal(1.0), cst.Float)
  assert isinstance(convert_to_cst_literal(1), cst.Integer)


def test_injector_fw_utils_get_import_root_name() -> None:
  """Verify `get_import_root` handles injector fw utils get import root name as expected."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import get_import_root, is_docstring

  assert get_import_root(cst.Name("foo")) == "foo"
  assert is_docstring(cst.SimpleStatementLine([cst.Expr(cst.SimpleString('""'))]), 0) is True


def test_injector_fw_utils_is_future_import() -> None:
  """Verify `is_future_import` handles injector fw utils is future import as expected."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import is_future_import

  assert (
    is_future_import(
      cst.SimpleStatementLine(
        [cst.ImportFrom(module=cst.Name("__future__"), names=[cst.ImportAlias(name=cst.Name("pi"))])]
      )
    )
    is True
  )


def test_injector_fw_utils_get_import_root_attr() -> None:
  """Verify `get_import_root` handles injector fw utils get import root attr as expected."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import get_import_root, is_docstring

  assert get_import_root(cst.Attribute(value=cst.Name("foo"), attr=cst.Name("bar"))) == "foo"
  assert is_docstring(cst.SimpleStatementLine([cst.Expr(cst.SimpleString('""'))]), 1) is False


def test_injector_fw_utils_convert_negative() -> None:
  """Verify `convert_to_cst_literal` handles injector fw utils convert negative as expected."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal

  neg_int = convert_to_cst_literal(-1)
  assert isinstance(neg_int, cst.UnaryOperation)
  assert isinstance(neg_int.operator, cst.Minus)
  assert isinstance(neg_int.expression, cst.Integer)
  assert neg_int.expression.value == "1"
  assert cst.Module([]).code_for_node(neg_int) == "-1"

  neg_float = convert_to_cst_literal(-1.0)
  assert isinstance(neg_float, cst.UnaryOperation)
  assert isinstance(neg_float.operator, cst.Minus)
  assert isinstance(neg_float.expression, cst.Float)
  assert neg_float.expression.value == "1.0"
  assert cst.Module([]).code_for_node(neg_float) == "-1.0"


def test_injector_fw_utils_convert_empty_containers() -> None:
  """Verifies converting empty list, tuple, and dict to CST literals."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal

  empty_list_node: cst.BaseExpression = convert_to_cst_literal([])
  assert isinstance(empty_list_node, cst.List)
  assert len(empty_list_node.elements) == 0

  empty_tuple_node: cst.BaseExpression = convert_to_cst_literal(())
  assert isinstance(empty_tuple_node, cst.Tuple)
  assert len(empty_tuple_node.elements) == 0

  empty_dict_node: cst.BaseExpression = convert_to_cst_literal({})
  assert isinstance(empty_dict_node, cst.Dict)
  assert len(empty_dict_node.elements) == 0
