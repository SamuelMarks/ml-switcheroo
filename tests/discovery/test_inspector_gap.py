import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.inspector import ApiInspector
import inspect


def test_inspector_cache():
  inspector = ApiInspector()
  inspector._package_cache["dummy"] = MagicMock(members={})
  res = inspector.inspect("dummy")
  assert res == {}


@patch("ml_switcheroo.discovery.inspector.griffe.load")
def test_inspector_griffe_module_recurse(mock_griffe):
  inspector = ApiInspector()
  mock_obj = MagicMock()
  mock_obj.members = {}

  mock_module = MagicMock()
  mock_module.is_alias = False
  mock_module.is_function = False
  mock_module.is_class = False
  mock_module.is_attribute = False
  mock_module.is_module = True
  mock_module.members = {}

  mock_obj.members["child_mod"] = mock_module
  mock_griffe.return_value = mock_obj

  res = inspector.inspect("dummy")
  assert res == {}


@patch("ml_switcheroo.discovery.inspector.griffe.load")
def test_inspector_griffe_exception(mock_griffe):
  inspector = ApiInspector()
  mock_obj = MagicMock()
  mock_obj.members = {}

  mock_bad = MagicMock()
  mock_bad.is_alias = False
  type(mock_bad).is_function = property(lambda self: (_ for _ in ()).throw(Exception("Test Error")))

  mock_obj.members["bad"] = mock_bad
  mock_griffe.return_value = mock_obj

  res = inspector.inspect("dummy")
  assert res == {}


@patch("ml_switcheroo.discovery.inspector.griffe.load", side_effect=Exception("Griffe Fail"))
@patch("ml_switcheroo.discovery.inspector.importlib.import_module")
def test_inspector_import_error(mock_import, mock_griffe):
  mock_import.side_effect = ImportError("Import Fail")
  inspector = ApiInspector()
  res = inspector.inspect("dummy_missing")
  assert res == {}


@patch("ml_switcheroo.discovery.inspector.griffe.load", side_effect=Exception("Griffe Fail"))
@patch("ml_switcheroo.discovery.inspector.importlib.import_module")
def test_inspector_other_error(mock_import, mock_griffe):
  mock_import.side_effect = Exception("Other Fail")
  inspector = ApiInspector()
  res = inspector.inspect("dummy_missing")
  assert res == {}


@patch("ml_switcheroo.discovery.inspector.griffe.load", side_effect=Exception("Griffe Fail"))
def test_inspector_runtime_recursion(mock_griffe):
  inspector = ApiInspector()

  class CyclicObj:
    pass

  obj1 = CyclicObj()
  obj2 = CyclicObj()
  obj1.child = obj2
  obj2.child = obj1

  class DepthObj:
    pass

  curr = DepthObj()
  for _ in range(7):
    next_obj = DepthObj()
    curr.child = next_obj
    curr = next_obj

  with patch("ml_switcheroo.discovery.inspector.importlib.import_module") as mock_import:

    class DummyModule:
      pass

    mock_module = DummyModule()
    mock_module.cyclic = obj1
    mock_module.depth = curr

    class BadMembers:
      @property
      def __class__(self):
        raise Exception("getmembers fail")

    # Bypass mock setattr issues by using simple object
    setattr(mock_module, "bad", BadMembers())
    setattr(mock_module, "_hidden", "hidden")

    class NestedModule:
      __package__ = "dummy"

    n_mod = NestedModule()
    setattr(mock_module, "nested", n_mod)

    mock_import.return_value = mock_module

    res = inspector.inspect("dummy")


@patch("ml_switcheroo.discovery.inspector.griffe.load", side_effect=Exception("Griffe Fail"))
def test_inspector_runtime_functions(mock_griffe):
  inspector = ApiInspector()

  with patch("ml_switcheroo.discovery.inspector.importlib.import_module") as mock_import:

    class DummyModule:
      __package__ = "dummy"

    mock_module = DummyModule()

    def good_func(a, *args):
      pass

    good_func.__module__ = "dummy.test"

    def bad_module_func():
      pass

    bad_module_func.__module__ = "other_lib"

    class CExtMock:
      __module__ = "dummy.c"
      pass

    setattr(mock_module, "f1", good_func)
    setattr(mock_module, "f2", bad_module_func)
    setattr(mock_module, "f3", CExtMock)
    setattr(mock_module, "c_int", 42)

    mock_import.return_value = mock_module
    res = inspector.inspect("dummy")

    assert "dummy.f1" in res
    assert res["dummy.f1"]["has_varargs"] == True
    assert res["dummy.f1"]["params"] == ["a", "args"]

    assert "dummy.f2" not in res

    assert "dummy.c_int" in res
    assert res["dummy.c_int"]["type"] == "attribute"


def test_extract_griffe_sig_attribute_error():
  inspector = ApiInspector()

  class BadFunc:
    name = "bad"

    @property
    def parameters(self):
      raise AttributeError("Fail")

  res = inspector._extract_griffe_sig(BadFunc(), "function")
  assert res["params"] == []


@patch("ml_switcheroo.discovery.inspector.inspect.signature", side_effect=ValueError("Test Error"))
def test_extract_runtime_sig_exceptions(mock_sig):
  inspector = ApiInspector()
  res = inspector._extract_runtime_sig(list, "list", "class")
  assert res["params"] == []
