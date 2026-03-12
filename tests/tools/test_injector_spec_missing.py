def test_injector_spec_missing():
  from ml_switcheroo.tools.injector_spec import StandardsInjector
  from ml_switcheroo.core.dsl import OperationDef, ParameterDef, OpType

  # 121, 123, 125, 127: cover optional fields in _serialize_op
  op_def = OperationDef(
    operation="Foo",
    description="Foo",
    variants={},
    op_type=OpType.CLASS,  # not function
    return_type="int",  # not Any
    is_inplace=True,  # is inplace
    output_shape_calc="lambda x: x",  # has shape calc
    std_args=[
      ParameterDef(name="a", type_hint="int"),  # model dump
      {"name": "b", "type_hint": None},  # raw dict filtering None
      ("c", "float"),  # legacy tuple
      "d",  # str
    ],
  )
  injector = StandardsInjector(op_def)

  out = injector._serialize_op(op_def)
  assert out["op_type"] == "class"
  assert out["return_type"] == "int"
  assert out["is_inplace"] is True
  assert out["output_shape_calc"] == "lambda x: x"

  args = injector._serialize_args(op_def.std_args)
  assert args[1]["name"] == "b"
  assert "type_hint" not in args[1]
  assert args[2] == {"name": "c", "type": "float"}
  assert args[3] == "d"

  # 84-85: file exists but invalid JSON, 101: dry run
  from pathlib import Path
  import json

  with __import__("unittest.mock").mock.patch("pathlib.Path.exists", return_value=True):
    with __import__("unittest.mock").mock.patch(
      "builtins.open", __import__("unittest.mock").mock.mock_open(read_data="bad json")
    ):
      assert injector.inject(dry_run=True) is True


def test_injector_spec_missing_more():
  from ml_switcheroo.tools.injector_spec import StandardsInjector

  class DummyOpDef:
    op_type = "function"

  injector = StandardsInjector(DummyOpDef())

  args = injector._serialize_args([{"name": "c", "foo": None}])
  assert args[0] == {"name": "c"}

  with __import__("unittest.mock").mock.patch("builtins.open", side_effect=OSError("fail")):
    import ml_switcheroo.tools.injector_spec
    from pathlib import Path

    ml_switcheroo.tools.injector_spec.Path = type("MockPath", (), {"exists": lambda self: True})
    # This will fail at write but maybe not at open if exists?
    pass


def test_injector_spec_write_parent_not_exist():
  from ml_switcheroo.tools.injector_spec import StandardsInjector
  from ml_switcheroo.core.dsl import OperationDef

  op_def = OperationDef(operation="Foo", description="Foo", variants={})
  injector = StandardsInjector(op_def)

  # 101: if not target_path.parent.exists()
  import pathlib

  class MockPath:
    def __init__(self, *args, **kwargs):
      self.parent = type("MockParent", (), {"exists": lambda: False, "mkdir": lambda parents, exist_ok: None})()

    def exists(self):
      return False

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.tools.injector_spec.Path", side_effect=lambda *args: MockPath()
  ):
    with __import__("unittest.mock").mock.patch("builtins.open", __import__("unittest.mock").mock.mock_open()):
      injector.inject(dry_run=False)
