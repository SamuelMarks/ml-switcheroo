def test_injector_fw_utils_missing():
  import libcst as cst
  from ml_switcheroo.tools.injector_fw.utils import (
    get_import_root,
    is_docstring,
    is_future_import,
    convert_to_cst_literal,
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

  # 124, 129, 139 - test conversion fallbacks/failures
  with __import__("unittest.mock").mock.patch("libcst.parse_expression", side_effect=Exception("fail")):
    assert isinstance(convert_to_cst_literal("something"), cst.SimpleString)

  class UnrecognizedObject:
    pass

  assert isinstance(convert_to_cst_literal(UnrecognizedObject()), cst.SimpleString)


def test_injector_fw_utils_convert_complex():
  import libcst as cst
  from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal

  assert isinstance(convert_to_cst_literal([1, 2]), cst.List)
  assert isinstance(convert_to_cst_literal((1, 2)), cst.Tuple)
  assert isinstance(convert_to_cst_literal({"a": 1, "b": 2}), cst.Dict)
  assert isinstance(convert_to_cst_literal(None), cst.Name)
  assert convert_to_cst_literal(None).value == "None"
  assert isinstance(convert_to_cst_literal(True), cst.Name)
  assert convert_to_cst_literal(True).value == "True"
  assert isinstance(convert_to_cst_literal(1.0), cst.Float)
  assert isinstance(convert_to_cst_literal(1), cst.Integer)


def test_injector_fw_utils_get_import_root_name():
  import libcst as cst
  from ml_switcheroo.tools.injector_fw.utils import get_import_root, is_docstring

  assert get_import_root(cst.Name("foo")) == "foo"

  assert is_docstring(cst.SimpleStatementLine([cst.Expr(cst.SimpleString('""'))]), 0) is True


def test_injector_fw_utils_is_future_import():
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


def test_injector_fw_utils_get_import_root_attr():
  import libcst as cst
  from ml_switcheroo.tools.injector_fw.utils import get_import_root, is_docstring

  assert get_import_root(cst.Attribute(value=cst.Name("foo"), attr=cst.Name("bar"))) == "foo"

  assert is_docstring(cst.SimpleStatementLine([cst.Expr(cst.SimpleString('""'))]), 1) is False


def test_injector_fw_utils_convert_negative():
  import libcst as cst
  from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal

  assert isinstance(convert_to_cst_literal(-1), cst.UnaryOperation)
  assert isinstance(convert_to_cst_literal(-1.0), cst.UnaryOperation)
