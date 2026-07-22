"""Test suite for the Plugin Injector Logic module."""

import pytest
from ml_switcheroo.core.dsl import PluginScaffoldDef, PluginType, Rule, LogicOp
from ml_switcheroo.tools.injector_plugin import PluginGenerator


@pytest.fixture
def plugin_dir(tmp_path):
  """Provides a mock plugin directory for testing."""
  d = tmp_path / "plugins"
  d.mkdir()
  return d


def test_generate_logic_operators(plugin_dir):
  """Generates logic operators."""
  gen = PluginGenerator(plugin_dir)
  rules = [
    Rule(if_arg="size", op=LogicOp.GT, val=512, use_api="large_algo"),
    Rule(if_arg="mode", op=LogicOp.IN, val=["a", "b"], use_api="ab_mode"),
    Rule(if_arg="count", op=LogicOp.NEQ, val=0, use_api="non_zero"),
  ]
  scaffold = PluginScaffoldDef(name="logic_hook", type=PluginType.CALL, doc="Ops", rules=rules)
  gen.generate(scaffold)
  content = (plugin_dir / "logic_hook.py").read_text("utf-8")
  assert 'val_0 = _get_kwarg_value(node, "size")' in content
  assert "if val_0 is not None and val_0 > 512:" in content
  assert "elif val_1 in ['a', 'b']:" in content
  assert "elif val_2 != 0:" in content
  assert '_create_dotted_name("large_algo")' in content
  assert '_create_dotted_name("ab_mode")' in content


def test_generate_logic_generation_check(plugin_dir):
  """Generates logic generation check."""
  gen = PluginGenerator(plugin_dir)
  r1 = [Rule(if_arg="x", op=LogicOp.EQ, val=1, use_api="one")]
  s1 = PluginScaffoldDef(name="update_test_v1", type=PluginType.CALL, doc="v1", rules=r1)
  gen.generate(s1)
  c1 = (plugin_dir / "update_test_v1.py").read_text("utf-8")
  assert "== 1" in c1
  r2 = [Rule(if_arg="x", op=LogicOp.LT, val=5, use_api="small")]
  s2 = PluginScaffoldDef(name="update_test_v2", type=PluginType.CALL, doc="v2", rules=r2)
  gen.generate(s2)
  c2 = (plugin_dir / "update_test_v2.py").read_text("utf-8")
  assert "< 5" in c2
  assert "== 1" not in c2
