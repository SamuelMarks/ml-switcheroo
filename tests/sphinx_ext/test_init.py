"""Test suite for the Init module."""

import typing

import os
from unittest import mock

from pydantic import JsonValue

from ml_switcheroo.sphinx_ext import setup


class EventCallbackProtocol(typing.Protocol):
  """Protocol for Sphinx event callbacks."""

  __name__: str


class MockApp:
  """Test helper class MockApp."""

  def __init__(self) -> None:
    """Initializes the MockApp instance."""
    self.directives: typing.Dict[str, type] = {}
    self.css_files: typing.List[str] = []
    self.js_files: typing.List[typing.Tuple[typing.Optional[str], typing.Dict[str, JsonValue]]] = []
    self.events: typing.List[typing.Tuple[str, EventCallbackProtocol]] = []

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

  def add_js_file(self, filename: typing.Optional[str], **kwargs: JsonValue) -> None:
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
  """Verifies the behavior of setup build all."""
  app = MockApp()
  result = setup(app)  # type: ignore[arg-type]
  assert result["version"]
  assert result["parallel_read_safe"] is True
  assert result["parallel_write_safe"] is True
  assert "switcheroo_demo" in app.directives
  assert any(("codemirror.min.css" in css for css in app.css_files))
  assert "switcheroo_demo.css" in app.css_files
  assert any((js[0] and "codemirror.min.js" in js[0] for js in app.js_files))
  assert any((js[0] is None and js[1].get("body") for js in app.js_files))
  event_names: typing.List[str] = [e[0] for e in app.events]
  assert "builder-inited" in event_names
  assert "build-finished" in event_names
  connected_funcs: typing.List[str] = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" in connected_funcs


@mock.patch.dict(os.environ, clear=True)
def test_setup_default_no_docs() -> None:
  """Verifies the behavior of setup default no documentation."""
  app = MockApp()
  setup(app)  # type: ignore[arg-type]
  connected_funcs: typing.List[str] = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" not in connected_funcs
