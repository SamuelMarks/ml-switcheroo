from ml_switcheroo.cli.handlers.define import (
  _resolve_inferred_apis,
  _inject_hub,
  _inject_spokes,
  _scaffold_plugins,
  _generate_test_file,
  handle_define,
)
from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant, PluginScaffoldDef
from unittest.mock import patch, MagicMock
from pathlib import Path


@patch("ml_switcheroo.cli.handlers.define._inject_hub")
def test_define_inject_hub_failure(MockHub, tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text(
    "name: Abs\noperation: abs\ndescription: ''\nvariants: {torch: {name: torch.abs}}\ntier: MATH\nstd_args: [x]"
  )

  MockHub.return_value = False
  assert handle_define(f, dry_run=False, no_test_gen=False) == 1


def test_define_yaml_is_none(tmp_path):
  import sys

  with patch.dict(sys.modules, {"yaml": None}):
    if "ml_switcheroo.cli.handlers.define" in sys.modules:
      del sys.modules["ml_switcheroo.cli.handlers.define"]
    import ml_switcheroo.cli.handlers.define

    assert ml_switcheroo.cli.handlers.define.handle_define(tmp_path / "test.yaml") == 1


def test_define_list_with_empty_doc(tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text(
    "---\n---\n- name: Abs\n  operation: abs\n  description: ''\n  variants: {torch: {name: torch.abs}}\n  tier: MATH\n  std_args: [x]"
  )
  assert handle_define(f, dry_run=True, no_test_gen=True) == 0


def test_inject_spokes_invalid_adapter():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"invalid_fw": FrameworkVariant(name="invalid")},
    tier="MATH",
    std_args=["x"],
  )
  _inject_spokes(op)


def test_scaffold_plugins_real():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test")},
    scaffold_plugins=[PluginScaffoldDef(name="TestPlugin", logic="test")],
    tier="MATH",
    std_args=["x"],
  )
  with patch("ml_switcheroo.cli.handlers.define.PluginGenerator") as MockGen:
    _scaffold_plugins(op, dry_run=False)
    MockGen.return_value.generate.assert_called_once()


def test_scaffold_plugins_fallback_path():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test")},
    scaffold_plugins=[PluginScaffoldDef(name="TestPlugin", logic="test")],
    tier="MATH",
    std_args=["x"],
  )
  import ml_switcheroo.plugins

  with patch.object(ml_switcheroo.plugins, "__file__", None):
    with patch("ml_switcheroo.cli.handlers.define.PluginGenerator") as MockGen:
      _scaffold_plugins(op, dry_run=True)
      MockGen.return_value.generate.assert_not_called()


def test_generate_test_file_real():
  op = OperationDef(name="test", operation="test", description="", variants={}, tier="MATH", std_args=["x"])
  with patch("ml_switcheroo.cli.handlers.define.TestCaseGenerator") as MockGen:
    _generate_test_file(op, MagicMock(), dry_run=False)
    MockGen.return_value.generate.assert_called_once()


def test_generate_test_file_dry_run():
  op = OperationDef(name="test", operation="test", description="", variants={}, tier="MATH", std_args=["x"])
  _generate_test_file(op, MagicMock(), dry_run=True)


def test_define_validation_error(tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text(
    "name: Abs\noperation: abs\ndescription: ''\nvariants: {torch: {name: torch.abs, requires_plugin: True}}\ntier: MATH\nstd_args: [x]"
  )
  assert handle_define(f) == 1


def test_inject_spokes_valid_adapter():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test")},
    tier="MATH",
    std_args=["x"],
  )
  with patch("ml_switcheroo.cli.handlers.define.FrameworkInjector") as MockGen:
    _inject_spokes(op)


def test_generate_test_file_error_log():
  op = OperationDef(name="test", operation="test", description="", variants={}, tier="MATH", std_args=["x"])
  with patch("ml_switcheroo.cli.handlers.define.TestCaseGenerator") as MockGen:
    MockGen.return_value.generate.side_effect = Exception("error")
    _generate_test_file(op, MagicMock())


def test_inject_spokes_infer():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test", api="infer")},
    tier="MATH",
    std_args=["x"],
  )
  _inject_spokes(op)


def test_scaffold_plugins_exception():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={},
    scaffold_plugins=[PluginScaffoldDef(name="TestPlugin", logic="test")],
    tier="MATH",
    std_args=["x"],
  )
  with patch("ml_switcheroo.cli.handlers.define.PluginGenerator") as MockGen:
    MockGen.side_effect = Exception("error")
    _scaffold_plugins(op)
