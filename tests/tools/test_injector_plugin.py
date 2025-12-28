"""
Tests for the Plugin Scaffolder Tool.

Verifies:
1. Generation of new plugin files (Call/Block).
2. Compilation of declarative rules into Python logic.
3. **Preservation of User Logic**: Ensuring manual edits are kept when regenerating.
4. Robustness against syntax errors or existing file corruption.
5. **Filename Normalization**: Validates Snake Case conversion.
6. **Auto-Wire Generation**: Validates that auto_wire dicts are injected into source code.
"""

import pytest
from pathlib import Path
from ml_switcheroo.core.dsl import PluginScaffoldDef, PluginType, Rule
from ml_switcheroo.tools.injector_plugin import PluginGenerator

@pytest.fixture
def plugin_dir(tmp_path):
  """Creates a temporary plugins directory."""
  d = tmp_path / "plugins"
  d.mkdir()
  return d

def test_filename_normalization(plugin_dir):
  """
  Verify that PascalCase names are converted to snake_case filenames.
  """
  gen = PluginGenerator(plugin_dir)

  # Case 1: PascalCase
  scaffold = PluginScaffoldDef(name="MyCustomHook", type=PluginType.CALL, doc="Test")
  gen.generate(scaffold)
  assert (plugin_dir / "my_custom_hook.py").exists()
  assert not (plugin_dir / "MyCustomHook.py").exists()

  # Case 2: camelCase
  scaffold2 = PluginScaffoldDef(name="tensorOps", type=PluginType.CALL, doc="Test")
  gen.generate(scaffold2)
  assert (plugin_dir / "tensor_ops.py").exists()

  # Case 3: Already snake
  scaffold3 = PluginScaffoldDef(name="already_valid", type=PluginType.CALL, doc="Test")
  gen.generate(scaffold3)
  assert (plugin_dir / "already_valid.py").exists()

  # Verify internal hook name matches spec (NOT filename)
  content = (plugin_dir / "my_custom_hook.py").read_text("utf-8")
  assert '@register_hook("MyCustomHook")' in content
  assert "def MyCustomHook(" in content

def test_generate_call_plugin(plugin_dir):
  """Verify generating a standard call plugin."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="my_hook", type=PluginType.CALL, doc="Test Hook")

  created = gen.generate(scaffold)

  assert created is True
  file_path = plugin_dir / "my_hook.py"
  assert file_path.exists()

  content = file_path.read_text("utf-8")
  assert '@register_hook("my_hook")' in content
  assert "def my_hook(node: cst.Call" in content
  # Check updated template without spaces
  assert '"""\nTest Hook\n"""' in content
  assert "# TODO: Implement custom logic" in content

def test_generate_block_plugin(plugin_dir):
  """Verify generating a block plugin uses the correct template."""
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="loop_hook", type=PluginType.BLOCK, doc="Loop transform")

  created = gen.generate(scaffold)
  assert created is True

  content = (plugin_dir / "loop_hook.py").read_text("utf-8")
  assert "def loop_hook(node: cst.CSTNode" in content
  # Should not contain specific helpers unless rules are present
  assert "def _get_kwarg_value" not in content

def test_generate_creates_directory(tmp_path):
  """Verify it creates the plugin directory if missing."""
  missing_dir = tmp_path / "ghost_plugins"
  gen = PluginGenerator(missing_dir)
  scaffold = PluginScaffoldDef(name="test", type=PluginType.CALL, doc="d")

  created = gen.generate(scaffold)

  assert created is True
  assert missing_dir.exists()
  assert (missing_dir / "test.py").exists()

def test_generate_plugin_with_rules(plugin_dir):
  """
  Verify that plugins generated with declarative rules contain correct logic.
  Covers int, string, and bool value matching logic compilation.
  """
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

  # 1. Helper Logic
  assert "def _get_kwarg_value" in content
  assert "def _node_to_literal" in content
  assert "def _create_dotted_name" in content

  # 2. String Match
  assert 'val_0 = _get_kwarg_value(node, "mode")' in content
  assert "if val_0 == 'nearest':" in content
  assert 'new_func = _create_dotted_name("jax.image.resize_nearest")' in content

  # 3. Boolean Match (elif + repr(True))
  assert 'val_1 = _get_kwarg_value(node, "antialias")' in content
  assert "elif val_1 == True:" in content
  assert 'new_func = _create_dotted_name("jax.image.resize_antialias")' in content

  # 4. Integer Match
  assert 'val_2 = _get_kwarg_value(node, "count")' in content
  assert "elif val_2 == 0:" in content
  assert 'new_func = _create_dotted_name("jax.noop")' in content

  # 5. Return structure
  # return node.with_changes is inside the block
  assert "return node.with_changes(func=new_func)" in content
  # Final fallback return
  assert content.strip().endswith("return node")

def test_preserves_user_logic(plugin_dir):
  """
  Verify that if a user modifies the logic body, regenerating the plugin
  preserves that logic while updating the wrapper/metadata.
  """
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="custom_logic", type=PluginType.CALL, doc="Original Doc")

  # 1. Generate Initial
  gen.generate(scaffold)
  file_path = plugin_dir / "custom_logic.py"

  # 2. Simulate User Edit
  # User adds print statements and custom return
  # Using clean formatting for input simulation
  user_code = ''' 
import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext

@register_hook("custom_logic") 
def custom_logic(node: cst.Call, ctx: HookContext) -> cst.CSTNode: 
    """Old Docstring.""" 
    print("User Custom Logic") 
    return node.with_changes(func=cst.Name("hacked")) 
'''
  file_path.write_text(user_code.strip(), encoding="utf-8")

  # 3. Regenerate with NEW docstring
  new_scaffold = PluginScaffoldDef(name="custom_logic", type=PluginType.CALL, doc="Updated Docstring")
  gen.generate(new_scaffold)

  # 4. Verify Preservation
  content = file_path.read_text("utf-8")

  # Metadata updated?
  # Uses new template without spaces
  assert '"""\nUpdated Docstring\n"""' in content
  assert '    """\n    Plugin Hook: Updated Docstring\n    """' in content

  # User Logic Preserved?
  assert 'print("User Custom Logic")' in content
  assert 'cst.Name("hacked")' in content

  # Old docstring inside function should be gone (replaced by template)
  assert '"""Old Docstring."""' not in content

  # We expect 4 occurrences of triple quotes:
  # 2 for File Header
  # 2 for Function Docstring
  assert content.count('"""') == 4

def test_preserves_logic_with_complex_indentation(plugin_dir):
  """
  Verify indentation is handled correctly when extracting and reinjecting body.
  """
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="indent_test", type=PluginType.CALL, doc="Doc")
  gen.generate(scaffold)

  file_path = plugin_dir / "indent_test.py"
  user_code = """ 
@register_hook("indent_test") 
def indent_test(node, ctx): 
    if True: 
        print("Indented") 
    return node
"""
  file_path.write_text(user_code.strip(), encoding="utf-8")

  gen.generate(scaffold)
  content = file_path.read_text("utf-8")

  # Check structure validity
  assert "    if True:" in content
  assert "        print" in content

  # Ensure no double indentation or stripping
  # The generated file uses 4 spaces.
  assert "\n    if True:" in content

def test_user_logic_trumps_rules(plugin_dir):
  """
  Scenario: User has written custom logic. Updates specify generated rules.
  Expectation: User logic is preserved, rules are ignored (preservation priority).
  """
  gen = PluginGenerator(plugin_dir)
  scaffold = PluginScaffoldDef(name="priority_test", type=PluginType.CALL, doc="Doc")
  gen.generate(scaffold)

  file_path = plugin_dir / "priority_test.py"
  file_path.write_text(""" 
@register_hook("priority_test") 
def priority_test(node, ctx): 
    return "UserLogic" 
""")

  # Regenerate with rules
  rules = [Rule(if_arg="x", is_val=1, use_api="y")]
  rule_scaffold = PluginScaffoldDef(name="priority_test", type=PluginType.CALL, doc="Doc", rules=rules)

  gen.generate(rule_scaffold)
  content = file_path.read_text("utf-8")

  assert 'return "UserLogic"' in content
  assert "val_0 =" not in content  # Rule logic skipped

def test_overwrite_on_syntax_error(plugin_dir, capsys):
  """
  Scenario: Existing file has syntax error (unparseable).
  Expectation: Generator logs warning and overwrites with default scaffold.
  """
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
  """
  Scenario: scaffold_plugins entry contains 'auto_wire' dict.
  Expectation: Generated file includes `auto_wire={...}` in decorator.
  """
  gen = PluginGenerator(plugin_dir)

  auto_data = {
    "ops": {
      "TestOp": {
        "std_args": ["x"],
        "variants": {"jax": {"api": "foo", "requires_plugin": "rewired"}}
      }
    }
  }

  scaffold = PluginScaffoldDef(
    name="rewired",
    type=PluginType.CALL,
    doc="Auto Wired",
    auto_wire=auto_data
  )

  gen.generate(scaffold)

  file_path = plugin_dir / "rewired.py"
  content = file_path.read_text("utf-8")

  # Check Decorator Injection
  assert '@register_hook(trigger="rewired", auto_wire={' in content
  assert '"TestOp":' in content
  assert '"api": "foo"' in content
