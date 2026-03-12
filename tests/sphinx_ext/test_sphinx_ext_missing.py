import pytest
from unittest.mock import patch, MagicMock


def test_sphinx_registry_missing():
  from ml_switcheroo.sphinx_ext.registry import scan_registry

  with patch("ml_switcheroo.sphinx_ext.registry.get_adapter", return_value=None):
    with patch("ml_switcheroo.sphinx_ext.registry.available_frameworks", return_value=["dummy"]):
      scan_registry()

  # Mocking target placeholder
  with patch("ml_switcheroo.sphinx_ext.registry.available_frameworks", return_value=["dummy"]):
    with patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order", return_value=["other"]):
      with patch("ml_switcheroo.sphinx_ext.registry.get_adapter") as mock_get:
        m = MagicMock()
        m.get_tiered_examples.return_value = {"tier": "code"}
        m.inherits_from = None
        mock_get.return_value = m
        scan_registry()


def test_sphinx_directive():
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
      # Pass a hierarchy where dummy is not present, but we manually seed the grouped?
      # actually if we pass empty hierarchy, grouped is empty, group_name not in grouped, hits line 250!
      # if we pass hierarchy with "dummy", it appends "dummy" to "TestGroup". Then members is not empty.
      pass

  # Actually, we can just inject into grouped?
  # wait, grouped is locally instantiated as defaultdict(list)
  # we can't easily make it have a key but empty list unless we mock defaultdict or something.
  # What if we just call it and mock `grouped[group_name]`? We can't.
  # Let's mock dict or just let it be. What if `hierarchy` has a root but `FRAMEWORK_GROUPS.get(root)` is "TestGroup"?
  # If we patch `FRAMEWORK_GROUPS`, we can do that. But how to make `members` empty?
  # Maybe we can mock `list` or `sorted` to return an empty list? No, `sorted` returns roots.
  # We can just skip line 255 if it's practically impossible, or we can mock `grouped` by patching defaultdict.
  from collections import defaultdict

  with patch("ml_switcheroo.sphinx_ext.rendering.defaultdict") as mock_dd:
    # Return a defaultdict that behaves normally but we will patch GROUP_ORDER to include something that gets left empty
    # Wait, if it gets left empty, it's not even in grouped, so it hits line 250!
    # If we want line 255 (if not members), it MUST be in grouped but have an empty list.
    # How to do that?
    # grouped["TestGroup"] = []
    # So we can just create a real defaultdict, and pre-populate it!
    real_dd = defaultdict(list)
    real_dd["TestGroup"] = []
    mock_dd.return_value = real_dd
    with patch("ml_switcheroo.sphinx_ext.rendering.GROUP_ORDER", ["TestGroup"]):
      _render_primary_options({})
