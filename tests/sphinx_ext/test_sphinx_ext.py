"""Test suite for the Sphinx Ext Missing module."""

from unittest.mock import MagicMock, patch

from ml_switcheroo.enums import RuntimeTensorOrValue


def test_sphinx_registry_missing() -> None:
  """Verify `scan_registry` handles sphinx registry missing as expected."""
  from ml_switcheroo.sphinx_ext.registry import scan_registry

  with (
    patch("ml_switcheroo.sphinx_ext.registry.get_adapter", return_value=None),
    patch("ml_switcheroo.sphinx_ext.registry.available_frameworks", return_value=["dummy"]),
  ):
    res1 = scan_registry()
  assert isinstance(res1, tuple)
  assert res1 == ({}, "{}", "{}")
  with (
    patch("ml_switcheroo.sphinx_ext.registry.available_frameworks", return_value=["dummy"]),
    patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order", return_value=["other"]),
    patch("ml_switcheroo.sphinx_ext.registry.get_adapter") as mock_get,
  ):
    m: MagicMock = MagicMock()
    m.get_tiered_examples.return_value = {"tier": "code"}
    m.inherits_from = None
    mock_get.return_value = m
    res2 = scan_registry()
  assert isinstance(res2, tuple)
  assert len(res2) == 3
  assert "dummy" in res2[0]
  assert '"code"' in res2[1]


def test_sphinx_directive() -> None:
  """Verify `SwitcherooDemo` handles sphinx directive as expected."""
  from docutils.statemachine import StringList

  from ml_switcheroo.sphinx_ext.directive import SwitcherooDemo

  d: SwitcherooDemo = SwitcherooDemo(
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
  with (
    patch("ml_switcheroo.sphinx_ext.directive.scan_registry", return_value=({"h": []}, "{}", "{}")),
    patch("ml_switcheroo.sphinx_ext.directive.render_demo_html", return_value="<html></html>"),
  ):
    nodes = d.run()
  assert len(nodes) == 1
  assert "<html></html>" in str(nodes[0])
  from ml_switcheroo.sphinx_ext.rendering import _render_primary_options

  with (
    patch("ml_switcheroo.sphinx_ext.rendering.GROUP_ORDER", ["TestGroup"]),
    patch("ml_switcheroo.sphinx_ext.rendering.FRAMEWORK_GROUPS", {"dummy": "TestGroup"}),
  ):
    pass
  from collections import defaultdict

  with patch("ml_switcheroo.sphinx_ext.rendering.defaultdict") as mock_dd:
    real_dd: dict[str, RuntimeTensorOrValue] = defaultdict(list)
    real_dd["TestGroup"] = []
    mock_dd.return_value = real_dd
    with patch("ml_switcheroo.sphinx_ext.rendering.GROUP_ORDER", ["TestGroup"]):
      out = _render_primary_options({})
  assert isinstance(out, str)
  assert "<optgroup" in out
