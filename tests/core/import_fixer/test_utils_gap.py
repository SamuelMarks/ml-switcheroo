"""Test suite for the Utils Gap module."""

import libcst as cst
from ml_switcheroo.core.import_fixer.utils import (
  get_root_name,
  is_future_import,
  create_dotted_name,
  get_signature,
  is_docstring,
)


def test_get_root_name_unknown():
  """Gets root name unknown."""
  assert get_root_name(cst.Ellipsis()) == ""


def test_is_future_import_not_future():
  """Checks if is future import not future."""
  mod = cst.parse_module("from foo import bar")
  assert not is_future_import(mod.body[0])


def test_is_future_import_not_import_from():
  """Checks if is future import not import from."""
  mod = cst.parse_module("import foo")
  assert not is_future_import(mod.body[0])


def test_is_future_import_none_module():
  """Checks if is future import none module."""
  mod = cst.parse_module("from . import bar")
  assert not is_future_import(mod.body[0])


def test_is_future_import_true():
  """Checks if is future import true."""
  mod = cst.parse_module("from __future__ import print_function")
  assert is_future_import(mod.body[0])


def test_create_dotted_name():
  """Creates dotted name."""
  node = create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_get_signature():
  """Gets signature."""
  mod = cst.parse_module("import   foo")
  sig = get_signature(mod.body[0])
  assert sig == "import foo"


def test_is_docstring():
  """Checks if is docstring."""
  mod = cst.parse_module('"""doc"""\nimport foo')
  assert is_docstring(mod.body[0], 0)
  assert not is_docstring(mod.body[1], 1)


def test_is_docstring_not_zero():
  """Checks if is docstring not zero."""
  mod = cst.parse_module('import foo\n"""doc"""')
  assert not is_docstring(mod.body[1], 1)


def test_get_root_name_cst():
  """Gets root name cst."""
  assert get_root_name(cst.Name("torch")) == "torch"
  assert get_root_name(cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))) == "torch"


def test_is_docstring_not_expr():
  """Checks if is docstring not expr."""
  mod = cst.parse_module("import foo")
  assert not is_docstring(mod.body[0], 0)


def test_is_docstring_not_string():
  """Checks if is docstring not string."""
  mod = cst.parse_module("1")
  assert not is_docstring(mod.body[0], 0)
