"""Test suite for the Injector Plugin module."""

import pytest
from ml_switcheroo.core.dsl import PluginScaffoldDef, PluginType, Rule
from ml_switcheroo.tools.injector_plugin import PluginGenerator


@pytest.fixture
def plugin_dir(tmp_path):
  """Provides a mock plugin directory for testing."""
  d = tmp_path / "plugins"
  d.mkdir()
  return d


def test_filename_normalization(plugin_dir):
  """Verifies the behavior of filename normalization."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="MyCustomHook", type=PluginType.CALL, doc="Test")
  gen.generate(scaffold)
  assert (plugin_dir / "my_custom_hook.py").exists()
  assert not (plugin_dir / "MyCustomHook.py").exists()
  scaffold2 = PluginScaffoldDef(name="tensorOps", type=PluginType.CALL, doc="Test")
  gen.generate(scaffold2)
  assert (plugin_dir / "tensor_ops.py").exists()
  scaffold3 = PluginScaffoldDef(name="already_valid", type=PluginType.CALL, doc="Test")
  gen.generate(scaffold3)
  assert (plugin_dir / "already_valid.py").exists()
  content = (plugin_dir / "my_custom_hook.py").read_text("utf-8")
  assert '@register_hook("MyCustomHook")' in content
  assert "def MyCustomHook(" in content


def test_generate_call_plugin(plugin_dir):
  """Generates call plugin."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="my_hook", type=PluginType.CALL, doc="Test Hook")
  created = gen.generate(scaffold)
  assert created is True
  file_path = plugin_dir / "my_hook.py"
  assert file_path.exists()
  content = file_path.read_text("utf-8")
  assert '@register_hook("my_hook")' in content
  assert "def my_hook(node: cst.Call" in content
  assert '"""\nTest Hook\n"""' in content
  assert "# TODO: Implement custom logic" in content


def test_generate_block_plugin(plugin_dir):
  """Generates block plugin."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="loop_hook", type=PluginType.BLOCK, doc="Loop transform")
  created = gen.generate(scaffold)
  assert created is True
  content = (plugin_dir / "loop_hook.py").read_text("utf-8")
  assert "def loop_hook(node: cst.CSTNode" in content
  assert "def _get_kwarg_value" not in content


def test_generate_creates_directory(tmp_path):
  """Generates creates directory."""
  missing_dir = tmp_path / "ghost_plugins"
  gen = PluginGenerator(missing_dir)
  scaffold = PluginScaffoldDef(name="test", type=PluginType.CALL, doc="d")
  created = gen.generate(scaffold)
  assert created is True
  assert missing_dir.exists()
  assert (missing_dir / "test.py").exists()


def test_generate_plugin_with_rules(plugin_dir):
  """Generates plugin with rules."""
  gen = PluginGenerator(plugin_dir)
  rules = [
    Rule(if_arg="mode", is_val="nearest", use_api="jax.image.resize_nearest"),
    Rule(if_arg="antialias", is_val=True, use_api="jax.image.resize_antialias"),
    Rule(if_arg="count", is_val=0, use_api="jax.noop"),
  ]
  scaffold = PluginScaffoldDef(name="rule_hook", type=PluginType.CALL, doc="Rules", rules=rules)
  gen.generate(scaffold)
  file_path = plugin_dir / "rule_hook.py"
  content = file_path.read_text("utf-8")
  assert "def _get_kwarg_value" in content
  assert "def _node_to_literal" in content
  assert "def _create_dotted_name" in content
  assert 'val_0 = _get_kwarg_value(node, "mode")' in content
  assert "if val_0 == 'nearest':" in content
  assert 'new_func = _create_dotted_name("jax.image.resize_nearest")' in content
  assert 'val_1 = _get_kwarg_value(node, "antialias")' in content
  assert "if val_1 == True:" in content
  assert 'new_func = _create_dotted_name("jax.image.resize_antialias")' in content
  assert 'val_2 = _get_kwarg_value(node, "count")' in content
  assert "if val_2 == 0:" in content
  assert 'new_func = _create_dotted_name("jax.noop")' in content
  assert "return node.with_changes(func=new_func)" in content
  assert content.strip().endswith("return node")


def test_preserves_user_logic(plugin_dir):
  """Verifies the behavior of preserves user logic."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="custom_logic", type=PluginType.CALL, doc="Original Doc")
  gen.generate(scaffold)
  file_path = plugin_dir / "custom_logic.py"
  user_code = '\nimport libcst as cst\nfrom ml_switcheroo.core.hooks import register_hook, HookContext\n\n@register_hook("custom_logic")\ndef custom_logic(node: cst.Call, ctx: HookContext) -> cst.CSTNode:\n    """Old Docstring."""\n    print("User Custom Logic")\n    return node.with_changes(func=cst.Name("hacked"))\n'
  file_path.write_text(user_code.strip(), encoding="utf-8")
  new_scaffold = PluginScaffoldDef(name="custom_logic", type=PluginType.CALL, doc="Updated Docstring")
  gen.generate(new_scaffold)
  content = file_path.read_text("utf-8")
  assert '"""\nUpdated Docstring\n"""' in content
  assert '    """\n    Plugin Hook: Updated Docstring\n    """' in content
  assert 'print("User Custom Logic")' in content
  assert 'cst.Name("hacked")' in content
  assert '"""Old Docstring."""' not in content
  assert content.count('"""') == 4


def test_preserves_logic_with_complex_indentation(plugin_dir):
  """Verifies the behavior of preserves logic with complex indentation."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="indent_test", type=PluginType.CALL, doc="Doc")
  gen.generate(scaffold)
  file_path = plugin_dir / "indent_test.py"
  user_code = '\n@register_hook("indent_test")\ndef indent_test(node, ctx):\n    if True:\n        print("Indented")\n    return node\n'
  file_path.write_text(user_code.strip(), encoding="utf-8")
  gen.generate(scaffold)
  content = file_path.read_text("utf-8")
  assert "    if True:" in content
  assert "        print" in content
  assert "\n    if True:" in content


def test_preserves_logic_with_simple_statement_suite(plugin_dir):
  """Test preserving logic from a single-line function body (SimpleStatementSuite)."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="simple_stmt", type=PluginType.CALL, doc="Doc")
  file_path = plugin_dir / "simple_stmt.py"
  # Single line body: def simple_stmt(node, ctx): pass
  user_code = '\n@register_hook("simple_stmt")\ndef simple_stmt(node, ctx): pass\n'
  file_path.write_text(user_code.strip(), encoding="utf-8")
  gen.generate(scaffold)
  content = file_path.read_text("utf-8")
  assert "pass" in content
  # It gets converted to an IndentedBlock
  assert "    pass" in content


def test_preserves_logic_empty_body(plugin_dir):
  """Test preserving logic when body becomes empty after docstring strip."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="empty_body", type=PluginType.CALL, doc="New Doc")
  file_path = plugin_dir / "empty_body.py"
  user_code = '\n@register_hook("empty_body")\ndef empty_body(node, ctx):\n    """Old Docstring."""\n'
  file_path.write_text(user_code.strip(), encoding="utf-8")
  gen.generate(scaffold)
  content = file_path.read_text("utf-8")
  assert "return node" in content


def test_injector_plugin_edge_cases(plugin_dir):
  """Test injector plugin edge cases for body extraction."""
  from ml_switcheroo.tools.injector_plugin import PluginGenerator

  gen = PluginGenerator(plugin_dir)

  # Cover line 244 and 283 by overriding cst.parse_module
  import libcst as cst
  import pytest

  with pytest.MonkeyPatch().context() as m:
    m.setattr(
      cst,
      "parse_module",
      lambda x: cst.Module(
        body=[
          cst.FunctionDef(
            name=cst.Name("temp"), params=cst.Parameters(), body=cst.SimpleStatementSuite(body=[cst.Pass()])
          )
        ]
      ),
    )
    assert gen._generate_cst_body_logic([]) == []
    from ml_switcheroo.core.dsl import Rule

    assert gen._generate_cst_body_logic([Rule(if_arg="x", is_val=1, use_api="y")]) == []


def test_user_logic_trumps_rules(plugin_dir):
  """Verifies the behavior of user logic trumps rules."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="priority_test", type=PluginType.CALL, doc="Doc")
  gen.generate(scaffold)
  file_path = plugin_dir / "priority_test.py"
  file_path.write_text('\n@register_hook("priority_test")\ndef priority_test(node, ctx):\n    return "UserLogic"\n')
  rules = [Rule(if_arg="x", is_val=1, use_api="y")]
  rule_scaffold = PluginScaffoldDef(name="priority_test", type=PluginType.CALL, doc="Doc", rules=rules)
  gen.generate(rule_scaffold)
  content = file_path.read_text("utf-8")
  assert 'return "UserLogic"' in content
  assert "val_0 =" not in content


def test_overwrite_on_syntax_error(plugin_dir, capsys):
  """Verifies the behavior of overwrite on syntax correctly handling an error."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="broken_file", type=PluginType.CALL, doc="Doc")
  file_path = plugin_dir / "broken_file.py"
  file_path.write_text("def broken_file(node, ctx): \n  syntax error here >>>", encoding="utf-8")
  gen.generate(scaffold)
  captured = capsys.readouterr()
  assert "Failed to parse existing plugin" in captured.out
  assert "Overwriting" in captured.out
  content = file_path.read_text("utf-8")
  assert "syntax error" not in content
  assert "# TODO: Implement custom logic" in content


def test_auto_wire_generation(plugin_dir):
  """Verifies the behavior of auto wire generation."""
  gen = PluginGenerator(plugin_dir)
  auto_data = {"ops": {"TestOp": {"std_args": ["x"], "variants": {"jax": {"api": "foo", "requires_plugin": "rewired"}}}}}
  scaffold = PluginScaffoldDef(name="rewired", type=PluginType.CALL, doc="Auto Wired", auto_wire=auto_data)
  gen.generate(scaffold)
  file_path = plugin_dir / "rewired.py"
  content = file_path.read_text("utf-8")
  assert '@register_hook(trigger="rewired", auto_wire={' in content
  assert '"TestOp":' in content
  assert '"api": "foo"' in content
