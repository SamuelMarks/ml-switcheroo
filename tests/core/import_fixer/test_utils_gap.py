"""Test suite for the Utils Gap module."""

import libcst as cst
import typing
from ml_switcheroo.core.import_fixer.utils import (
  get_root_name,
  is_future_import,
  create_dotted_name,
  get_signature,
  is_docstring,
)


def test_get_root_name_unknown() -> None:
  """Gets root name unknown."""
  assert get_root_name(cst.Ellipsis()) == ""


def test_is_future_import_not_future() -> None:
  """Checks if is future import not future."""
  mod: cst.Module = cst.parse_module("from foo import bar")
  assert not is_future_import(mod.body[0])  # type: ignore


def test_is_future_import_not_import_from() -> None:
  """Checks if is future import not import from."""
  mod: cst.Module = cst.parse_module("import foo")
  assert not is_future_import(mod.body[0])  # type: ignore


def test_is_future_import_none_module() -> None:
  """Checks if is future import none module."""
  mod: cst.Module = cst.parse_module("from . import bar")
  assert not is_future_import(mod.body[0])  # type: ignore


def test_is_future_import_true() -> None:
  """Checks if is future import true."""
  mod: cst.Module = cst.parse_module("from __future__ import print_function")
  assert is_future_import(mod.body[0])  # type: ignore


def test_create_dotted_name() -> None:
  """Creates dotted name."""
  node: typing.Any = create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_get_signature() -> None:
  """Gets signature."""
  mod: cst.Module = cst.parse_module("import   foo")
  sig: str = get_signature(mod.body[0])
  assert sig == "import foo"


def test_is_docstring() -> None:
  """Checks if is docstring."""
  mod: cst.Module = cst.parse_module('"""doc"""\nimport foo')
  assert is_docstring(mod.body[0], 0)
  assert not is_docstring(mod.body[1], 1)


def test_is_docstring_not_zero() -> None:
  """Checks if is docstring not zero."""
  mod: cst.Module = cst.parse_module('import foo\n"""doc"""')
  assert not is_docstring(mod.body[1], 1)


def test_get_root_name_cst() -> None:
  """Gets root name cst."""
  assert get_root_name(cst.Name("torch")) == "torch"
  assert get_root_name(cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))) == "torch"


def test_is_docstring_not_expr() -> None:
  """Checks if is docstring not expr."""
  mod: cst.Module = cst.parse_module("import foo")
  assert not is_docstring(mod.body[0], 0)


def test_is_docstring_not_string() -> None:
  """Checks if is docstring not string."""
  mod: cst.Module = cst.parse_module("1")
  assert not is_docstring(mod.body[0], 0)
