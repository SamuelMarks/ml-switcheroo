import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.sphinx_ext.directive import SwitcherooDemo
from ml_switcheroo.sphinx_ext.registry import scan_registry
from ml_switcheroo.sphinx_ext.rendering import render_demo_html


def test_directive_run(monkeypatch):
  monkeypatch.setattr("ml_switcheroo.sphinx_ext.directive.scan_registry", lambda: ({}, "{}", "{}"))
  monkeypatch.setattr("ml_switcheroo.sphinx_ext.directive.render_demo_html", lambda *args: "<div></div>")

  directive = SwitcherooDemo(
    name="test",
    arguments=[],
    options={},
    content=[],
    lineno=1,
    content_offset=0,
    block_text="",
    state=MagicMock(),
    state_machine=MagicMock(),
  )
  res = directive.run()
  assert len(res) == 1
  assert "raw" in str(type(res[0]))


def test_registry_scan_registry(monkeypatch):
  import ml_switcheroo.frameworks.base as fb
  from ml_switcheroo.sphinx_ext import registry

  # force line 41 continue
  class EmptyAdapter:
    @property
    def import_alias(self):
      return None

    @property
    def get_tiered_examples(self):
      return lambda: {}

  def mock_get(fw):
    return EmptyAdapter()

  monkeypatch.setattr(fb, "get_adapter", mock_get)
  # mock priority order to trigger 122-123 and 126
  # For 122-123, it tries to remove "source_placeholder" from candidates or similar.
  # Actually wait, I'll just use a python patch for scan_registry.
  # We want line 122-123.
  # It's probably `candidates.remove(src_fw)`. If it raises ValueError, it sets `tgt_fw = candidates[0]`.

  monkeypatch.setattr("ml_switcheroo.config.get_framework_priority_order", lambda: ["fw1", "fw2"])

  class FakeAdapter:
    import_alias = ("mod", "alias")

    def get_tiered_examples(self):
      return {"tier1": {"example": "foo"}}

  monkeypatch.setattr(fb, "get_adapter", lambda fw: FakeAdapter())
  monkeypatch.setattr(fb, "available_frameworks", lambda: ["fw1", "fw2"])

  res = registry.scan_registry()

  # what if candidates is empty? (line 126)
  monkeypatch.setattr("ml_switcheroo.config.get_framework_priority_order", lambda: ["fw1"])
  monkeypatch.setattr(fb, "available_frameworks", lambda: ["fw1"])
  res = registry.scan_registry()


def test_rendering_missing(monkeypatch):
  import ml_switcheroo.sphinx_ext.rendering as rendering

  # line 66, 71, 72
  # def_source = priority_order[0] if priority_order else "source_placeholder"
  monkeypatch.setattr("ml_switcheroo.config.get_framework_priority_order", lambda: [])
  rendering.render_demo_html({}, "{}", "{}")

  monkeypatch.setattr("ml_switcheroo.config.get_framework_priority_order", lambda: ["fw1"])
  rendering.render_demo_html({}, "{}", "{}")

  # line 260 continue?
  hierarchy = {"fw1": {"framework_meta": {"tiers": []}}}
  monkeypatch.setattr("ml_switcheroo.config.get_framework_priority_order", lambda: ["fw1", "fw2"])
  rendering.render_demo_html(hierarchy, "{}", "{}")
