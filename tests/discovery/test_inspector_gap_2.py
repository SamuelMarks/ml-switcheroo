import pytest
import types
import inspect
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.inspector import ApiInspector


@patch("ml_switcheroo.discovery.inspector.griffe.load", side_effect=Exception("Griffe Fail"))
def test_inspector_runtime_recursion_depth_cycle(mock_griffe):
  inspector = ApiInspector()

  M1 = types.ModuleType("M1")
  M1.__package__ = "dummy"
  M2 = types.ModuleType("M2")
  M2.__package__ = "dummy"

  M1.m2 = M2
  M2.m1 = M1

  class C0:
    __module__ = "dummy"

  class C1:
    __module__ = "dummy"

  class C2:
    __module__ = "dummy"

  class C3:
    __module__ = "dummy"

  class C4:
    __module__ = "dummy"

  class C5:
    __module__ = "dummy"

  class C6:
    __module__ = "dummy"

  C0.c1 = C1
  C1.c2 = C2
  C2.c3 = C3
  C3.c4 = C4
  C4.c5 = C5
  C5.c6 = C6

  with patch("ml_switcheroo.discovery.inspector.importlib.import_module") as mock_import:
    mock_module = types.ModuleType("dummy")
    mock_module.__package__ = "dummy"
    mock_module.m1 = M1
    mock_module.c0 = C0

    real_getmembers = inspect.getmembers
    with patch("ml_switcheroo.discovery.inspector.inspect.getmembers") as mock_gm:

      def side_effect(obj):
        if obj == M2:
          raise Exception("Fail")
        return real_getmembers(obj)

      mock_gm.side_effect = side_effect

      mock_import.return_value = mock_module
      res = inspector.inspect("dummy")
      assert "dummy.c0.c1.c2.c3.c4.c5" in res
      assert "dummy.c0.c1.c2.c3.c4.c5.c6" not in res


@patch("ml_switcheroo.discovery.inspector.griffe.load", side_effect=Exception("Griffe Fail"))
def test_inspector_runtime_c_ext_fallback(mock_griffe):
  inspector = ApiInspector()
  with patch("ml_switcheroo.discovery.inspector.importlib.import_module") as mock_import:
    mock_module = types.ModuleType("dummy")
    mock_module.__package__ = "dummy"

    class MockFunc:
      __module__ = "dummy"

    mock_func = MockFunc()

    # Mock func needs to be something inspect.isfunction() or isclass() etc.
    # Let's use a real function
    def mock_f():
      pass

    mock_f.__module__ = "dummy"

    setattr(mock_module, "f1", mock_f)

    with patch.object(inspector, "_extract_runtime_sig", side_effect=Exception("Sig Error")):
      mock_import.return_value = mock_module
      res = inspector.inspect("dummy")
      assert res["dummy.f1"]["params"] == ["x"]


def test_extract_griffe_sig_attribute_error_coverage():
  inspector = ApiInspector()

  class BadParam:
    @property
    def kind(self):
      raise AttributeError("bad kind")

    @property
    def name(self):
      return "p"

  class BadFunc:
    name = "bad"
    parameters = [BadParam()]

  res = inspector._extract_griffe_sig(BadFunc(), "function")
  assert res["params"] == []
