import pytest
import libcst as cst
from ml_switcheroo.core.import_fixer.utils import (
  get_root_name,
  is_future_import,
  create_dotted_name,
  get_signature,
  is_docstring,
)


def test_get_root_name_unknown():
  # Pass a string instead of Name or Attribute
  assert get_root_name(cst.Ellipsis()) == ""


def test_is_future_import_not_future():
  # An ImportFrom without module name or a module name that is not __future__
  mod = cst.parse_module("from foo import bar")
  assert not is_future_import(mod.body[0])


def test_is_future_import_not_import_from():
  mod = cst.parse_module("import foo")
  assert not is_future_import(mod.body[0])


def test_is_future_import_none_module():
  mod = cst.parse_module("from . import bar")
  assert not is_future_import(mod.body[0])


def test_is_future_import_true():
  mod = cst.parse_module("from __future__ import print_function")
  assert is_future_import(mod.body[0])


def test_create_dotted_name():
  node = create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_get_signature():
  mod = cst.parse_module("import   foo")
  sig = get_signature(mod.body[0])
  assert sig == "import foo"


def test_is_docstring():
  mod = cst.parse_module('"""doc"""\nimport foo')
  assert is_docstring(mod.body[0], 0)
  assert not is_docstring(mod.body[1], 1)


def test_is_docstring_not_zero():
  mod = cst.parse_module('import foo\n"""doc"""')
  assert not is_docstring(mod.body[1], 1)


def test_get_root_name_cst():
  assert get_root_name(cst.Name("torch")) == "torch"
  assert get_root_name(cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))) == "torch"


def test_is_docstring_not_expr():
  mod = cst.parse_module("import foo")
  assert not is_docstring(mod.body[0], 0)


def test_is_docstring_not_string():
  mod = cst.parse_module("1")
  assert not is_docstring(mod.body[0], 0)
