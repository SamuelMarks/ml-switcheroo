from ml_switcheroo.cli.handlers.define import (
  _resolve_inferred_apis,
  _inject_hub,
  _inject_spokes,
  _scaffold_plugins,
  _generate_test_file,
)
from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant, PluginScaffoldDef
from unittest.mock import patch, MagicMock


def test_resolve_inferred_apis():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test", api="infer")},
    tier="MATH",
    std_args=["x"],
  )
  with patch("ml_switcheroo.cli.handlers.define.SimulatedReflection") as MockRef:
    MockRef.return_value.discover.return_value = "torch.inferred"
    _resolve_inferred_apis(op)
    assert op.variants["torch"].api == "torch.inferred"

    MockRef.return_value.discover.return_value = None
    op.variants["torch"].api = "infer"
    _resolve_inferred_apis(op)
    assert op.variants["torch"].api == "infer"


def test_inject_hub_error():
  op = OperationDef(name="test", operation="test", description="", variants={}, tier="MATH", std_args=["x"])
  with patch("ml_switcheroo.cli.handlers.define.StandardsInjector") as MockInj:
    MockInj.side_effect = Exception("error")
    assert not _inject_hub(op)


def test_inject_spokes_error():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test")},
    tier="MATH",
    std_args=["x"],
  )
  with patch("ml_switcheroo.cli.handlers.define.FrameworkInjector") as MockInj:
    MockInj.side_effect = Exception("error")
    _inject_spokes(op)


def test_scaffold_plugins_error():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test", requires_plugin="TestPlugin")},
    tier="MATH",
    std_args=["x"],
  )
  with patch("ml_switcheroo.cli.handlers.define.PluginGenerator") as MockGen:
    MockGen.side_effect = Exception("error")
    _scaffold_plugins(op)


def test_scaffold_plugins_success():
  op = OperationDef(
    name="test",
    operation="test",
    description="",
    variants={"torch": FrameworkVariant(name="torch.test", requires_plugin="TestPlugin")},
    tier="MATH",
    std_args=["x"],
  )
  with patch("ml_switcheroo.cli.handlers.define.PluginGenerator") as MockGen:
    MockGen.return_value.generate.return_value = True
    _scaffold_plugins(op)


def test_generate_test_file_error():
  op = OperationDef(name="test", operation="test", description="", variants={}, tier="MATH", std_args=["x"])
  with patch("ml_switcheroo.cli.handlers.define.TestCaseGenerator") as MockGen:
    MockGen.side_effect = Exception("error")
    _generate_test_file(op, MagicMock())
