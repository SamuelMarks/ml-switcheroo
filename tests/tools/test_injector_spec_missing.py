"""Test suite for the Injector Spec Missing module."""


def test_injector_spec_missing():
  """Verifies the behavior of injector spec missing."""
  from ml_switcheroo.tools.injector_spec import StandardsInjector
  from ml_switcheroo.core.dsl import OperationDef, ParameterDef, OpType

  op_def = OperationDef(
    operation="Foo",
    description="Foo",
    variants={},
    op_type=OpType.CLASS,
    return_type="int",
    is_inplace=True,
    output_shape_calc="lambda x: x",
    std_args=[ParameterDef(name="a", type_hint="int"), {"name": "b", "type_hint": None}, ("c", "float"), "d"],
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
  with __import__("unittest.mock").mock.patch("pathlib.Path.exists", return_value=True):
    with __import__("unittest.mock").mock.patch(
      "builtins.open", __import__("unittest.mock").mock.mock_open(read_data="bad json")
    ):
      assert injector.inject(dry_run=True) is True


def test_injector_spec_missing_more():
  """Verifies the behavior of injector spec missing more."""
  from ml_switcheroo.tools.injector_spec import StandardsInjector

  class DummyOpDef:
    """Dummy Op Def class for testing purposes."""

    op_type = "function"

  injector = StandardsInjector(DummyOpDef())
  args = injector._serialize_args([{"name": "c", "foo": None}])
  assert args[0] == {"name": "c"}
  with __import__("unittest.mock").mock.patch("builtins.open", side_effect=OSError("fail")):
    import ml_switcheroo.tools.injector_spec

    ml_switcheroo.tools.injector_spec.Path = type("MockPath", (), {"exists": lambda self: True})
    pass


def test_injector_spec_write_parent_not_exist():
  """Verifies the behavior of injector spec write parent not exist."""
  from ml_switcheroo.tools.injector_spec import StandardsInjector
  from ml_switcheroo.core.dsl import OperationDef

  op_def = OperationDef(operation="Foo", description="Foo", variants={})
  injector = StandardsInjector(op_def)

  class MockPath:
    """Mock Path class for testing purposes."""

    def __init__(self, *args, **kwargs):
      """Initializes the MockPath instance."""
      self.parent = type(
        "MockParent", (), {"exists": lambda self: False, "mkdir": lambda self, parents=False, exist_ok=False: None}
      )()

    def exists(self):
      """Mock implementation of exists."""
      return False

    def __truediv__(self, other):
      return self

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.tools.injector_spec.resolve_semantics_dir", return_value=MockPath()
  ):
    with __import__("unittest.mock").mock.patch("builtins.open", __import__("unittest.mock").mock.mock_open()):
      injector.inject(dry_run=False)
