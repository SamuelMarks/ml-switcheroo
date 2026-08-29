"""Tests for docs/conf.py."""

import inspect
import os

# Avoid actual execution of the configuration logic on import
# We will use runpy to test the module scope
import runpy
import sys
import types
from pathlib import Path
from typing import Callable, Dict, Optional

import pytest

docs_dir: Path = Path(__file__).parent.parent / "docs"


def test_conf_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  # Test without BUILD_ALL_DOCS
  monkeypatch.delenv("BUILD_ALL_DOCS", raising=False)
  namespace: Dict[str, list[str]] = runpy.run_path(str(docs_dir / "conf.py"))

  assert "sphinx.ext.autodoc" in namespace["extensions"]
  assert "autoapi.extension" not in namespace["extensions"]

  # Test with BUILD_ALL_DOCS
  monkeypatch.setenv("BUILD_ALL_DOCS", "1")
  namespace2: Dict[str, list[str]] = runpy.run_path(str(docs_dir / "conf.py"))

  assert "autoapi.extension" in namespace2["extensions"]


def test_linkcode_resolve_non_py_domain() -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  assert linkcode_resolve("cpp", {"module": "test"}) is None


def test_linkcode_resolve_no_module() -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  assert linkcode_resolve("py", {"module": ""}) is None


def test_linkcode_resolve_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  monkeypatch.setitem(sys.modules, "missing_module", None)  # Ensure missing
  assert linkcode_resolve("py", {"module": "missing_module"}) is None


def test_linkcode_resolve_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  mock_mod: types.ModuleType = types.ModuleType("test_mod")
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "missing_attr"}) is None


def test_linkcode_resolve_unwrap_and_inspect_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  mock_mod: types.ModuleType = types.ModuleType("test_mod")

  # Setup an object with __wrapped__ that throws TypeError on inspect
  class DummyObj:
    """Dummy obj."""

    pass

  inner_obj: DummyObj = DummyObj()
  outer_obj: DummyObj = DummyObj()
  outer_obj.__wrapped__ = inner_obj

  mock_mod.my_attr = outer_obj
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  def mock_getsourcefile(obj: DummyObj) -> str:
    """Mock getsourcefile."""
    raise TypeError()

  monkeypatch.setattr(inspect, "getsourcefile", mock_getsourcefile)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_no_source_file(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  mock_mod: types.ModuleType = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: None)
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_outside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  mock_mod: types.ModuleType = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  # Mock inspect to return a path that resolves outside the repo (e.g. system lib)
  monkeypatch.setattr(inspect, "getsourcefile", lambda o: "/usr/lib/python3.9/os.py")
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None


def test_linkcode_resolve_success(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  mock_mod: types.ModuleType = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  # Project root is docs_dir.parent
  project_root: Path = docs_dir.parent
  mock_file: Path = project_root / "src" / "test_mod.py"

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: str(mock_file))
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line1", "line2"], 10))

  url: Optional[str] = linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"})

  assert url == "https://github.com/SamuelMarks/ml-switcheroo/blob/master/src/test_mod.py#L10-L11"


def test_linkcode_resolve_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  namespace: Dict[str, Callable[[str, Dict[str, str]], Optional[str]]] = runpy.run_path(str(docs_dir / "conf.py"))
  linkcode_resolve: Callable[[str, Dict[str, str]], Optional[str]] = namespace["linkcode_resolve"]

  mock_mod: types.ModuleType = types.ModuleType("test_mod")
  mock_mod.my_attr = "some_val"
  monkeypatch.setitem(sys.modules, "test_mod", mock_mod)

  monkeypatch.setattr(inspect, "getsourcefile", lambda o: "C:\\Windows\\System32")
  monkeypatch.setattr(inspect, "getsourcelines", lambda o: (["line"], 1))

  def mock_relpath(path: str, start: str) -> str:
    """Mock relpath."""
    raise ValueError("Paths on different drives")

  monkeypatch.setattr(os.path, "relpath", mock_relpath)

  assert linkcode_resolve("py", {"module": "test_mod", "fullname": "my_attr"}) is None
