"""Tests for docs/conf.py."""

import os
import sys
import types
import inspect
from pathlib import Path

# Avoid actual execution of the configuration logic on import
# We will use runpy to test the module scope
import runpy

docs_dir = Path(__file__).parent.parent / "docs"


def test_conf_evaluation(monkeypatch):
  """Tests that conf.py evaluates successfully in both full and partial modes."""
  # Test without BUILD_ALL_DOCS
  monkeypatch.delenv("BUILD_ALL_DOCS", raising=False)
  namespace = runpy.run_path(str(docs_dir / "conf.py"))

  assert "sphinx.ext.autodoc" in namespace["extensions"]
  assert "autoapi.extension" not in namespace["extensions"]

  # Test with BUILD_ALL_DOCS
  monkeypatch.setenv("BUILD_ALL_DOCS", "1")
  namespace2 = runpy.run_path(str(docs_dir / "conf.py"))

  assert "autoapi.extension" in namespace2["extensions"]


def test_linkcode_resolve_non_py_domain():
  """Tests linkcode_resolve returns None for non-python domains."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  assert linkcode_resolve("cpp", {"module": "test"}) is None


def test_linkcode_resolve_no_module():
  """Tests linkcode_resolve returns None when module is empty."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  assert linkcode_resolve("py", {"module": ""}) is None


def test_linkcode_resolve_module_not_found(monkeypatch):
  """Tests linkcode_resolve returns None when module is not in sys.modules."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  monkeypatch.setitem(sys.modules, "missing_module", None)  # Ensure missing
  assert linkcode_resolve("py", {"module": "missing_module"}) is None


def test_linkcode_resolve_attribute_error(monkeypatch):
  """Tests linkcode_resolve returns None when attribute is missing."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  mock_mod = types.ModuleType("test_mod")
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "missing_attr"}) is None


def test_linkcode_resolve_unwrap_and_inspect_error(monkeypatch):
  """Tests linkcode_resolve unwrap logic and handling of inspect errors."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  mock_mod = types.ModuleType("test_mod")

  # Setup an object with __wrapped__ that throws TypeError on inspect
  class DummyObj:
    """Dummy obj."""

    pass

  inner_obj = DummyObj()
  outer_obj = DummyObj()
  outer_obj.__wrapped__ = inner_obj

  mock_mod.my_attr = outer_obj
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  def mock_getsourcefile(obj):
    """Mock getsourcefile."""
    raise TypeError()

  monkeypatch.setattr(inspect, "getsourcefile", mock_getsourcefile)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_no_source_file(monkeypatch):
  """Tests linkcode_resolve returns None when source file is missing."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  mock_mod = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: None)
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_outside_repo(monkeypatch):
  """Tests linkcode_resolve returns None for files outside repo."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  mock_mod = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  # Mock inspect to return a path that resolves outside the repo (e.g. system lib)
  monkeypatch.setattr(inspect, "getsourcefile", lambda o: "/usr/lib/python3.9/os.py")
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_success(monkeypatch):
  """Tests successful GitHub URL resolution."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  mock_mod = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  # Project root is docs_dir.parent
  project_root = docs_dir.parent
  mock_file = project_root / "src" / "test_mod.py"

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: str(mock_file))
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line1", "line2"], 10))

  url = linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"})

  assert url == "https://github.com/SamuelMarks/ml-switcheroo/blob/master/src/test_mod.py#L10-L11"


def test_linkcode_resolve_value_error(monkeypatch):
  """Test linkcode_resolve handles ValueError during relpath calculation."""
  namespace = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve = namespace["linkcode_resolve"]

  mock_mod = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: "C:\\Windows\\System32")
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  def mock_relpath(path, start):
    """Mock relpath."""
    raise ValueError("Paths on different drives")

  monkeypatch.setattr(os.path, "relpath", mock_relpath)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None
