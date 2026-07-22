"""Test suite for the Sphinx Ext Missing module."""

from unittest.mock import patch, MagicMock


def test_sphinx_registry_missing():
  """Verifies the behavior of sphinx registry missing."""
  from ml_switcheroo.sphinx_ext.registry import scan_registry

  with patch("ml_switcheroo.sphinx_ext.registry.get_adapter", return_value=None):
    with patch("ml_switcheroo.sphinx_ext.registry.available_frameworks", return_value=["dummy"]):
      scan_registry()
  with patch("ml_switcheroo.sphinx_ext.registry.available_frameworks", return_value=["dummy"]):
    with patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order", return_value=["other"]):
      with patch("ml_switcheroo.sphinx_ext.registry.get_adapter") as mock_get:
        m = MagicMock()
        m.get_tiered_examples.return_value = {"tier": "code"}
        m.inherits_from = None
        mock_get.return_value = m
        scan_registry()


def test_sphinx_directive():
  """Verifies the behavior of sphinx directive."""
  from ml_switcheroo.sphinx_ext.directive import SwitcherooDemo
  from docutils.statemachine import StringList

  d = SwitcherooDemo(
    name="switcheroo_demo",
    arguments=[],
    options={},
    content=StringList([], source=""),
    lineno=1,
    content_offset=1,
    block_text="",
    state=MagicMock(),
    state_machine=MagicMock(),
  )
  with patch("ml_switcheroo.sphinx_ext.directive.scan_registry", return_value=({"h": []}, "{}", "{}")):
    with patch("ml_switcheroo.sphinx_ext.directive.render_demo_html", return_value="<html></html>"):
      d.run()
  from ml_switcheroo.sphinx_ext.rendering import _render_primary_options

  with patch("ml_switcheroo.sphinx_ext.rendering.GROUP_ORDER", ["TestGroup"]):
    with patch("ml_switcheroo.sphinx_ext.rendering.FRAMEWORK_GROUPS", {"dummy": "TestGroup"}):
      pass
  from collections import defaultdict

  with patch("ml_switcheroo.sphinx_ext.rendering.defaultdict") as mock_dd:
    real_dd = defaultdict(list)
    real_dd["TestGroup"] = []
    mock_dd.return_value = real_dd
    with patch("ml_switcheroo.sphinx_ext.rendering.GROUP_ORDER", ["TestGroup"]):
      _render_primary_options({})
