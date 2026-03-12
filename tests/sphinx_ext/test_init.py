import os
from unittest import mock

import pytest
from ml_switcheroo.sphinx_ext import setup


class MockApp:
  def __init__(self):
    self.directives = {}
    self.css_files = []
    self.js_files = []
    self.events = []

  def add_directive(self, name, directive):
    self.directives[name] = directive

  def add_css_file(self, filename):
    self.css_files.append(filename)

  def add_js_file(self, filename, **kwargs):
    self.js_files.append((filename, kwargs))

  def connect(self, event, callback):
    self.events.append((event, callback))


def test_setup_default():
  app = MockApp()

  if "HOMEPAGE_ONLY" in os.environ:
    del os.environ["HOMEPAGE_ONLY"]

  result = setup(app)

  assert result["version"]
  assert result["parallel_read_safe"] is True
  assert result["parallel_write_safe"] is True

  assert "switcheroo_demo" in app.directives

  # Check that CSS files are added
  assert any("codemirror.min.css" in css for css in app.css_files)
  assert "switcheroo_demo.css" in app.css_files

  # Check that JS files are added
  assert any(js[0] and "codemirror.min.js" in js[0] for js in app.js_files)
  assert any(js[0] is None and js[1].get("body") for js in app.js_files)

  # Check events
  event_names = [e[0] for e in app.events]
  assert "builder-inited" in event_names
  assert "build-finished" in event_names

  # generate_op_docs should be connected
  connected_funcs = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" in connected_funcs


@mock.patch.dict(os.environ, {"HOMEPAGE_ONLY": "1"})
def test_setup_homepage_only():
  app = MockApp()
  setup(app)

  connected_funcs = [e[1].__name__ for e in app.events]
  assert "generate_op_docs" not in connected_funcs
