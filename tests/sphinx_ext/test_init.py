"""Test suite for the Init module."""

import os
import typing
from unittest import mock

from pydantic import JsonValue
from sphinx.application import Sphinx

from ml_switcheroo.sphinx_ext import setup


class EventCallbackProtocol(typing.Protocol):
  """Protocol for Sphinx event callbacks."""

  __name__: str


class MockApp:
  """Test double `MockApp` implementing `add_directive`, `add_css_file` for init tests."""

  def __init__(self) -> None:
    """Initializes the MockApp instance."""
    self.directives: dict[str, type] = {}
    self.css_files: list[str] = []
    self.js_files: list[tuple[str | None, dict[str, JsonValue]]] = []
    self.events: list[tuple[str, EventCallbackProtocol]] = []
    self.sphinx_app: mock.MagicMock = mock.MagicMock(spec=Sphinx)
    self.sphinx_app.add_directive.side_effect = self.add_directive
    self.sphinx_app.add_css_file.side_effect = self.add_css_file
    self.sphinx_app.add_js_file.side_effect = self.add_js_file
    self.sphinx_app.connect.side_effect = self.connect

  def add_directive(self, name: str, directive: type) -> None:
    """Mock implementation of add directive.

    Args:
        name (str): Directive name.
        directive: Directive instance.
    """
    self.directives[name] = directive

  def add_css_file(self, filename: str) -> None:
    """Mock implementation of add css file.

    Args:
        filename (str): Filename string.
    """
    self.css_files.append(filename)

  def add_js_file(self, filename: str | None, **kwargs: JsonValue) -> None:
    """Mock implementation of add js file.

    Args:
        filename (Optional[str]): Filename string.
        **kwargs: Keyword arguments.
    """
    self.js_files.append((filename, kwargs))

  def connect(self, event: str, callback: EventCallbackProtocol) -> None:
    """Mock implementation of connect.

    Args:
        event (str): Event name.
        callback (Callable): Callback function.
    """
    self.events.append((event, callback))


@mock.patch.dict(os.environ, {"BUILD_ALL_DOCS": "1"})
def test_setup_build_all() -> None:
  """Verify `get` emits `switcheroo_demo` for setup build all."""
  app = MockApp()
  result = setup(app.sphinx_app)
  assert result["version"]
  assert result["parallel_read_safe"] is True
  assert result["parallel_write_safe"] is True
  assert "switcheroo_demo" in app.directives
  assert any(("codemirror.min.css" in css for css in app.css_files))
  assert "switcheroo_demo.css" in app.css_files
  assert any((js[0] and "codemirror.min.js" in js[0] for js in app.js_files))
  assert any((js[0] is None and js[1].get("body") for js in app.js_files))
  event_names: list[str] = [e[0] for e in app.events]
  assert "builder-inited" in event_names
  assert "build-finished" in event_names
  connected_funcs: list[str] = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" in connected_funcs


@mock.patch.dict(os.environ, clear=True)
def test_setup_default_no_docs() -> None:
  """Verify `MockApp` handles setup default no docs as expected."""
  app = MockApp()
  setup(app.sphinx_app)
  connected_funcs: list[str] = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" not in connected_funcs
