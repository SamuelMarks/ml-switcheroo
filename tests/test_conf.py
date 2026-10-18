"""Tests for docs/conf.py."""

import typing

import inspect
import os

# Avoid actual execution of the configuration logic on import
# We will use runpy to test the module scope
import runpy
import sys
import types
from pathlib import Path


docs_dir: typing.Any = Path(__file__).parent.parent / "docs"


def test_conf_evaluation(monkeypatch: typing.Any) -> None:
  """Verify conf evaluation behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  # Test without BUILD_ALL_DOCS
  monkeypatch.delenv("BUILD_ALL_DOCS", raising=False)
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))

  assert "sphinx.ext.autodoc" in namespace["extensions"]
  assert "autoapi.extension" not in namespace["extensions"]

  # Test with BUILD_ALL_DOCS
  monkeypatch.setenv("BUILD_ALL_DOCS", "1")
  namespace2: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))

  assert "autoapi.extension" in namespace2["extensions"]


def test_linkcode_resolve_non_py_domain() -> None:
  """Verify linkcode resolve non py domain behavior."""
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  assert linkcode_resolve("cpp", {"module": "test"}) is None


def test_linkcode_resolve_no_module() -> None:
  """Verify linkcode resolve no module behavior."""
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  assert linkcode_resolve("py", {"module": ""}) is None


def test_linkcode_resolve_module_not_found(monkeypatch: typing.Any) -> None:
  """Verify linkcode resolve module not found behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  monkeypatch.setitem(sys.modules, "missing_module", None)  # Ensure missing
  assert linkcode_resolve("py", {"module": "missing_module"}) is None


def test_linkcode_resolve_attribute_error(monkeypatch: typing.Any) -> None:
  """Verify linkcode resolve attribute error behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  mock_mod: typing.Any = types.ModuleType("test_mod")
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "missing_attr"}) is None


def test_linkcode_resolve_unwrap_and_inspect_error(monkeypatch: typing.Any) -> None:
  """Verify linkcode resolve unwrap and inspect error behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  mock_mod: typing.Any = types.ModuleType("test_mod")

  # Setup an object with __wrapped__ that throws TypeError on inspect
  class DummyObj:
    """Dummy obj."""

    pass

  inner_obj: typing.Any = DummyObj()
  outer_obj: typing.Any = DummyObj()
  outer_obj.__wrapped__ = inner_obj

  mock_mod.my_attr = outer_obj
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  def mock_getsourcefile(obj: typing.Any) -> typing.Any:
    """Mock getsourcefile.

    Args:
        obj: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    raise TypeError()

  monkeypatch.setattr(inspect, "getsourcefile", mock_getsourcefile)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_no_source_file(monkeypatch: typing.Any) -> None:
  """Verify linkcode resolve no source file behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  mock_mod: typing.Any = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: None)
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_outside_repo(monkeypatch: typing.Any) -> None:
  """Verify linkcode resolve outside repo behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  mock_mod: typing.Any = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  # Mock inspect to return a path that resolves outside the repo (e.g. system lib)
  monkeypatch.setattr(inspect, "getsourcefile", lambda o: "/usr/lib/python3.9/os.py")
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_success(monkeypatch: typing.Any) -> None:
  """Verify linkcode resolve success behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  mock_mod: typing.Any = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  # Project root is docs_dir.parent
  project_root: typing.Any = docs_dir.parent
  mock_file: typing.Any = project_root / "src" / "test_mod.py"

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: str(mock_file))
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line1", "line2"], 10))

  url: typing.Any = linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"})

  assert url == "https://github.com/SamuelMarks/ml-switcheroo/blob/master/src/test_mod.py#L10-L11"


def test_linkcode_resolve_value_error(monkeypatch: typing.Any) -> None:
  """Verify linkcode resolve value error behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  namespace: typing.Any = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: typing.Any = namespace["linkcode_resolve"]

  mock_mod: typing.Any = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: "C:\\Windows\\System32")
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  def mock_relpath(path: typing.Any, start: typing.Any) -> typing.Any:
    """Mock relpath.

    Args:
        path: Test argument or fixture.
        start: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    raise ValueError("Paths on different drives")

  monkeypatch.setattr(os.path, "relpath", mock_relpath)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None
