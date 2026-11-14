"""Unit tests for static CST helper functions in ``ml_switcheroo.core.import_fixer.utils``."""

import libcst as cst

from ml_switcheroo.core.import_fixer.utils import (
  create_dotted_name,
  get_root_name,
  get_signature,
  is_docstring,
  is_future_import,
)


def test_get_root_name() -> None:
  """Verify ``get_root_name`` extracts root identifiers from names and nested attributes."""
  assert get_root_name(cst.Name("torch")) == "torch"

  attr2 = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert get_root_name(attr2) == "torch"

  attr3 = cst.Attribute(
    value=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")),
    attr=cst.Name("Module"),
  )
  assert get_root_name(attr3) == "torch"

  assert get_root_name(cst.Attribute(value=cst.Ellipsis(), attr=cst.Name("x"))) == ""
  assert get_root_name(cst.Pass()) == ""


def test_create_dotted_name() -> None:
  """Verify ``create_dotted_name`` builds ``cst.Name`` and nested ``cst.Attribute`` chains."""
  single = create_dotted_name("torch")
  assert isinstance(single, cst.Name)
  assert single.value == "torch"

  dotted = create_dotted_name("torch.nn.Module")
  assert isinstance(dotted, cst.Attribute)
  assert dotted.attr.value == "Module"
  assert isinstance(dotted.value, cst.Attribute)
  assert dotted.value.attr.value == "nn"
  assert isinstance(dotted.value.value, cst.Name)
  assert dotted.value.value.value == "torch"


def test_get_signature() -> None:
  """Verify ``get_signature`` unwraps statement lines and normalizes internal whitespace."""
  import_stmt = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])])
  assert get_signature(import_stmt) == "import torch"

  direct_import = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  assert get_signature(direct_import) == "import torch"

  spaced_mod = cst.parse_module("import   foo")
  assert get_signature(spaced_mod.body[0]) == "import foo"

  empty_line = cst.SimpleStatementLine(body=[])
  assert get_signature(empty_line) == "pass"


def test_is_docstring() -> None:
  """Verify ``is_docstring`` detects simple and concatenated string expressions at module index 0."""
  doc_node = cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString('"""Doc"""'))])
  assert is_docstring(doc_node, 0) is True
  assert is_docstring(doc_node, 1) is False

  concat_node = cst.SimpleStatementLine(
    body=[cst.Expr(value=cst.ConcatenatedString(left=cst.SimpleString('"A"'), right=cst.SimpleString('"B"')))]
  )
  assert is_docstring(concat_node, 0) is True

  non_str_expr = cst.SimpleStatementLine(body=[cst.Expr(value=cst.Name("x"))])
  assert is_docstring(non_str_expr, 0) is False

  int_mod = cst.parse_module("1")
  assert is_docstring(int_mod.body[0], 0) is False

  import_mod = cst.parse_module("import foo")
  assert is_docstring(import_mod.body[0], 0) is False

  multi_stmt_mod = cst.parse_module('"""doc"""; x = 1')
  assert is_docstring(multi_stmt_mod.body[0], 0) is False

  assert is_docstring(cst.Pass(), 0) is False


def test_is_future_import() -> None:
  """Verify ``is_future_import`` identifies ``from __future__ import ...`` directives."""
  future_mod = cst.parse_module("from __future__ import annotations")
  assert is_future_import(future_mod.body[0]) is True

  multi_with_future = cst.SimpleStatementLine(
    body=[
      cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))]),
      cst.ImportFrom(module=cst.Name("__future__"), names=[cst.ImportAlias(name=cst.Name("annotations"))]),
    ]
  )
  assert is_future_import(multi_with_future) is True

  regular_from = cst.parse_module("from foo import bar")
  assert is_future_import(regular_from.body[0]) is False

  dotted_from = cst.parse_module("from foo.bar import baz")
  assert is_future_import(dotted_from.body[0]) is False

  relative_from = cst.parse_module("from . import bar")
  assert is_future_import(relative_from.body[0]) is False

  regular_import = cst.parse_module("import foo")
  assert is_future_import(regular_import.body[0]) is False

  multi_non_future = cst.SimpleStatementLine(
    body=[
      cst.ImportFrom(module=cst.Name("sys"), names=[cst.ImportAlias(name=cst.Name("path"))]),
      cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(name=cst.Name("path"))]),
      cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))]),
    ]
  )
  assert is_future_import(multi_non_future) is False

  assert is_future_import(cst.Pass()) is False
