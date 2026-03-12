import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import importlib

from ml_switcheroo.core.hooks import (
  HookContext,
  register_hook,
  get_hook,
  get_all_hook_metadata,
  clear_hooks,
  load_plugins,
  _import_from_dir,
  _HOOKS,
  _HOOK_METADATA,
)
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.schema import PluginTraits


class MockSymbol:
  def __init__(self, name):
    self.name = name


class MockSymbolTable:
  def __init__(self, type_mapping):
    self.type_mapping = type_mapping

  def get_type(self, node):
    return self.type_mapping.get(node)


def test_resolve_type():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  # Context without symbol table
  ctx = HookContext(semantics=None, config=config, symbol_table=None)
  assert ctx.resolve_type("node") is None

  # Context with symbol table
  sym_table = MockSymbolTable(
    {
      "node_tensor": MockSymbol("MyTensor"),
      "node_module": MockSymbol("SomeModule"),
      "node_other": MockSymbol("OtherType"),
      "node_none": None,
    }
  )
  ctx = HookContext(semantics=None, config=config, symbol_table=sym_table)

  assert ctx.resolve_type("node_tensor") == "Tensor"
  assert ctx.resolve_type("node_module") == "Module"
  assert ctx.resolve_type("node_other") == "OtherType"
  assert ctx.resolve_type("node_none") is None
  assert ctx.resolve_type("missing_node") is None


def test_plugin_traits_cases():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=config)

  # 1. conf has empty plugin_traits -> return default PluginTraits() (Line 136)
  semantics.get_framework_config.return_value = {"plugin_traits": {}}
  traits = ctx.plugin_traits
  assert isinstance(traits, PluginTraits)

  # 2. conf has unhandled type for plugin_traits -> return default PluginTraits() (Line 143)
  semantics.get_framework_config.return_value = {"plugin_traits": "some_string"}
  traits = ctx.plugin_traits
  assert isinstance(traits, PluginTraits)


def test_current_variant_no_semantics_or_op_id():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  ctx = HookContext(semantics=None, config=config)
  assert ctx.current_variant is None

  ctx = HookContext(semantics=MagicMock(), config=config)
  ctx.current_op_id = None
  assert ctx.current_variant is None


def test_current_variant_valid():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "jax.some_op", "plugin": "some_plugin"}
  ctx = HookContext(semantics=semantics, config=config)
  ctx.current_op_id = "some_op"

  variant = ctx.current_variant
  assert variant is not None
  assert variant.api == "jax.some_op"


def test_lookup_api_no_semantics():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = HookContext(semantics=None, config=config)
  assert ctx.lookup_api("some_op") is None


def test_lookup_signature_no_semantics():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = HookContext(semantics=None, config=config)
  assert ctx.lookup_signature("some_op") == []


def test_lookup_signature_dict_param():
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = MagicMock()
  semantics.get_definition_by_id.return_value = {
    "std_args": ["arg1", ["arg2", "default"], {"name": "arg3", "type": "int"}, {"no_name": "here"}]
  }
  ctx = HookContext(semantics=semantics, config=config)
  args = ctx.lookup_signature("some_op")
  assert args == ["arg1", "arg2", "arg3"]


def test_register_hook_invalid_autowire(capsys):
  clear_hooks()

  @register_hook("my_trigger", auto_wire={"ops": "invalid_because_string"})
  def my_hook(node, ctx):
    pass

  captured = capsys.readouterr()
  assert "Invalid auto_wire spec" in captured.out
  assert "my_trigger" in _HOOKS
  assert "my_trigger" not in _HOOK_METADATA


def test_register_hook_valid_autowire():
  clear_hooks()

  @register_hook("my_trigger_valid", auto_wire={"ops": {"op1": {"api": "test"}}})
  def my_hook(node, ctx):
    pass

  assert "my_trigger_valid" in _HOOKS
  assert "my_trigger_valid" in _HOOK_METADATA


def test_get_hook_lazy_load():
  clear_hooks()
  with patch("ml_switcheroo.core.hooks.load_plugins") as mock_load:
    get_hook("missing")
    mock_load.assert_called_once()


def test_get_all_hook_metadata_lazy_load():
  clear_hooks()
  with patch("ml_switcheroo.core.hooks.load_plugins") as mock_load:
    get_all_hook_metadata()
    mock_load.assert_called_once()


def test_load_plugins_exception_default(capsys):
  clear_hooks()
  orig_import = __import__

  def mock_import(name, *args, **kwargs):
    if name == "ml_switcheroo.plugins":
      raise ImportError("Mocked error")
    return orig_import(name, *args, **kwargs)

  with patch("builtins.__import__", side_effect=mock_import):
    load_plugins()

  # Should not crash, just print error
  captured = capsys.readouterr()
  assert "Failed to auto-load default plugins" in captured.out


def test_load_plugins_extra_dirs(tmp_path):
  clear_hooks()
  extra_dir = tmp_path / "extra_plugins"
  extra_dir.mkdir()

  # Mock _import_from_dir to just return 1
  with patch("ml_switcheroo.core.hooks._import_from_dir", return_value=1) as mock_import:
    total = load_plugins(extra_dirs=[extra_dir])
    assert total >= 1
    mock_import.assert_called_with(extra_dir, base_package=None)


def test_import_from_dir_package_success(tmp_path):
  clear_hooks()
  plugin_dir = tmp_path / "plugins"
  plugin_dir.mkdir()

  (plugin_dir / "__init__.py").write_text("")
  (plugin_dir / "my_plugin.py").write_text("def init(): pass")

  with patch("importlib.import_module") as mock_import:
    count = _import_from_dir(plugin_dir, base_package="some.package")
    mock_import.assert_called_once_with("some.package.my_plugin")
    assert count == 1


def test_import_from_dir_package_fail_fallback(tmp_path):
  clear_hooks()
  plugin_dir = tmp_path / "plugins"
  plugin_dir.mkdir()

  # Create an __init__.py so it could be a package, but we test fallback
  (plugin_dir / "__init__.py").write_text("")
  (plugin_dir / "my_plugin.py").write_text("def init(): pass")

  # Mock import_module to raise ImportError, simulating package load failure
  with patch("importlib.import_module", side_effect=ImportError("Mocked error")):
    count = _import_from_dir(plugin_dir, base_package="some.package")

  assert count == 1  # Should succeed via fallback


def test_import_from_dir_exception(tmp_path, capsys):
  clear_hooks()
  plugin_dir = tmp_path / "plugins"
  plugin_dir.mkdir()

  # Create invalid python file
  (plugin_dir / "my_bad_plugin.py").write_text("this is not python")

  count = _import_from_dir(plugin_dir, base_package=None)
  assert count == 0
  captured = capsys.readouterr()
  assert "Failed to load plugin my_bad_plugin.py" in captured.out


def test_hooks_gap_141():
  from ml_switcheroo.core.hooks import HookContext
  from ml_switcheroo.semantics.schema import PluginTraits

  traits = PluginTraits()
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.semantics.manager import SemanticsManager

  sem = type("MockSem", (SemanticsManager,), {"get_framework_config": lambda self, fw: {"plugin_traits": traits}})()
  ctx = HookContext(semantics=sem, config=RuntimeConfig())
  ctx.target_fw = "b"
  res = ctx.plugin_traits
  assert type(res) is type(traits)
  print("RES ID:", id(res), "TRAITS ID:", id(traits))
  assert id(res) == id(traits)
  print("RES IS", type(res), type(traits))
