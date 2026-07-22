"""Test suite for the Init module."""

import os
from unittest import mock
from ml_switcheroo.sphinx_ext import setup


class MockApp:
  """Mock App class for testing purposes."""

  def __init__(self):
    """Initializes the MockApp instance."""
    self.directives = {}
    self.css_files = []
    self.js_files = []
    self.events = []

  def add_directive(self, name, directive):
    """Mock implementation of add directive."""
    self.directives[name] = directive

  def add_css_file(self, filename):
    """Mock implementation of add css file."""
    self.css_files.append(filename)

  def add_js_file(self, filename, **kwargs):
    """Mock implementation of add js file."""
    self.js_files.append((filename, kwargs))

  def connect(self, event, callback):
    """Mock implementation of connect."""
    self.events.append((event, callback))


@mock.patch.dict(os.environ, {"BUILD_ALL_DOCS": "1"})
def test_setup_build_all():
  """Verifies the behavior of setup build all."""
  app = MockApp()
  result = setup(app)
  assert result["version"]
  assert result["parallel_read_safe"] is True
  assert result["parallel_write_safe"] is True
  assert "switcheroo_demo" in app.directives
  assert any(("codemirror.min.css" in css for css in app.css_files))
  assert "switcheroo_demo.css" in app.css_files
  assert any((js[0] and "codemirror.min.js" in js[0] for js in app.js_files))
  assert any((js[0] is None and js[1].get("body") for js in app.js_files))
  event_names = [e[0] for e in app.events]
  assert "builder-inited" in event_names
  assert "build-finished" in event_names
  connected_funcs = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" in connected_funcs


@mock.patch.dict(os.environ, clear=True)
def test_setup_default_no_docs():
  """Verifies the behavior of setup default no documentation."""
  app = MockApp()
  setup(app)
  connected_funcs = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" not in connected_funcs
