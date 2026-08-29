"""Test suite for the Init module."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

from ml_switcheroo.sphinx_ext import setup


class MockApp:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockApp instance."""
    self.directives: Dict[str, Any] = {}
    self.css_files: List[str] = []
    self.js_files: List[Tuple[Optional[str], Dict[str, Any]]] = []
    self.events: List[Tuple[str, Callable]] = []

  def add_directive(self, name: str, directive: Any) -> None:
    """Mock implementation of add directive.

    Args:
        name (str): Directive name.
        directive (Any): Directive instance.
    """
    self.directives[name] = directive

  def add_css_file(self, filename: str) -> None:
    """Mock implementation of add css file.

    Args:
        filename (str): Filename string.
    """
    self.css_files.append(filename)

  def add_js_file(self, filename: Optional[str], **kwargs: Any) -> None:
    """Mock implementation of add js file.

    Args:
        filename (Optional[str]): Filename string.
        **kwargs (Any): Keyword arguments.
    """
    self.js_files.append((filename, kwargs))

  def connect(self, event: str, callback: Callable) -> None:
    """Mock implementation of connect.

    Args:
        event (str): Event name.
        callback (Callable): Callback function.
    """
    self.events.append((event, callback))


@mock.patch.dict(os.environ, {"BUILD_ALL_DOCS": "1"})
def test_setup_build_all() -> None:
  """Verifies the behavior of setup build all."""
  app: MockApp = MockApp()
  result: Dict[str, Any] = setup(app)  # type: ignore
  assert result["version"]
  assert result["parallel_read_safe"] is True
  assert result["parallel_write_safe"] is True
  assert "switcheroo_demo" in app.directives
  assert any(("codemirror.min.css" in css for css in app.css_files))
  assert "switcheroo_demo.css" in app.css_files
  assert any((js[0] and "codemirror.min.js" in js[0] for js in app.js_files))
  assert any((js[0] is None and js[1].get("body") for js in app.js_files))
  event_names: List[str] = [e[0] for e in app.events]
  assert "builder-inited" in event_names
  assert "build-finished" in event_names
  connected_funcs: List[str] = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" in connected_funcs


@mock.patch.dict(os.environ, clear=True)
def test_setup_default_no_docs() -> None:
  """Verifies the behavior of setup default no documentation."""
  app: MockApp = MockApp()
  setup(app)  # type: ignore
  connected_funcs: List[str] = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" not in connected_funcs
