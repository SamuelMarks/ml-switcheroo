"""Test suite for the Injector Spec module."""

import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.core.dsl import OperationDef, OpType
from ml_switcheroo.tools.injector_spec import StandardsInjector


@pytest.fixture
def sample_op() -> OperationDef:
  """Docstring."""
  return OperationDef(
    operation="LogSoftmax", description="Log Softmax.", op_type=OpType.FUNCTION, std_args=["x", "dim"], variants={}
  )


def test_injector_finds_correct_file(sample_op: OperationDef) -> None:
  """Verifies the behavior of injector finds correct file."""

  def check_tier(tier: SemanticTier, expected_file: str, op_override: Optional[OperationDef] = None) -> None:
    """Checks tier."""
    target_op: OperationDef = op_override if op_override else sample_op
    injector: StandardsInjector = StandardsInjector(target_op, tier=tier)
    with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
      mock_resolve.return_value = Path("/mock/semantics")
      m_open: MagicMock = mock_open(read_data="{}")
      with patch("builtins.open", m_open):
        with patch("pathlib.Path.mkdir"):
          with patch("pathlib.Path.exists", return_value=True):
            injector.inject(dry_run=False)
      expected_path: Path = Path(f"/mock/semantics/{expected_file}")
      args_list: List[Any] = [c.args[0] for c in m_open.call_args_list]
      assert expected_path in args_list

  check_tier(SemanticTier.NEURAL, "odl/LogSoftmax.yaml")
  extra_op2: OperationDef = OperationDef(operation="abs", description="util", std_args=[], variants={})
  check_tier(SemanticTier.ARRAY_API, "odl/abs.yaml", op_override=extra_op2)
  extra_op: OperationDef = OperationDef(operation="manual_utility", description="util", std_args=[], variants={})
  check_tier(SemanticTier.EXTRAS, "odl/manual_utility.yaml", op_override=extra_op)


def test_injector_appends_new_op(sample_op: OperationDef) -> None:
  """Verifies the behavior of injector appends new op."""
  injector: StandardsInjector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)
  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")
    m_open: MagicMock = mock_open()
    with patch("builtins.open", m_open):
      with patch("pathlib.Path.mkdir"):
        with patch("pathlib.Path.exists", return_value=False):
          injector.inject()
    handle: MagicMock = m_open()
    written_data: str = "".join((str(call.args[0]) for call in handle.write.call_args_list))
    data: Dict[str, Any] = yaml.safe_load(written_data)
    assert data["operation"] == "LogSoftmax"
    assert data["description"] == "Log Softmax."
    assert data["std_args"] == ["x", "dim"]


def test_injector_updates_existing_op(sample_op: OperationDef) -> None:
  """Verifies the behavior of injector updates existing op."""
  injector: StandardsInjector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)
  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")
    m_open: MagicMock = mock_open()
    with patch("builtins.open", m_open):
      with patch("pathlib.Path.mkdir"):
        with patch("pathlib.Path.exists", return_value=True):
          injector.inject()
    handle: MagicMock = m_open()
    written_data: str = "".join((str(call.args[0]) for call in handle.write.call_args_list))
    data: Dict[str, Any] = yaml.safe_load(written_data)
    assert data["operation"] == "LogSoftmax"
    assert data["description"] == "Log Softmax."
    assert len(data["std_args"]) == 2


def test_injector_dry_run(sample_op: OperationDef, capsys: pytest.CaptureFixture[str]) -> None:
  """Verifies the behavior of injector dry run."""
  injector: StandardsInjector = StandardsInjector(sample_op, tier=SemanticTier.NEURAL)
  with patch("ml_switcheroo.tools.injector_spec.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/mock")
    with patch("builtins.open", mock_open(read_data="{}")) as m_open:
      with patch("pathlib.Path.exists", return_value=True):
        injector.inject(dry_run=True)
      handle: MagicMock = m_open()
      handle.write.assert_not_called()
      captured: pytest.CaptureResult[str] = capsys.readouterr()
      assert "[Dry Run]" in captured.out
      assert "LogSoftmax" in captured.out


# --- Merged from test_injector_spec_missing2.py ---


def test_injector_spec_write_parent_not_exist() -> None:
  """Verifies the behavior of injector spec write parent not exist."""
  from ml_switcheroo.core.dsl import OperationDef
  from ml_switcheroo.tools.injector_spec import StandardsInjector

  op_def: OperationDef = OperationDef(operation="Foo", description="Foo", variants={})
  injector: StandardsInjector = StandardsInjector(op_def)
  import tempfile

  with tempfile.TemporaryDirectory() as td:
    p: pathlib.Path = pathlib.Path(td) / "nested" / "file.json"
    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.tools.injector_spec.resolve_semantics_dir", return_value=p
    ):
      injector.inject(dry_run=False)
      assert p.exists()


# --- Merged from test_injector_spec_extra_loop.py ---


def test_injector_spec_legacy_tuple_args() -> None:
  """Verifies that legacy tuple arguments are properly converted."""
  # Pass self as None
  clean_args: list[Any] = StandardsInjector._serialize_args(None, args=[("x", "int"), ("y",)])  # type: ignore[arg-type]
  assert clean_args == [{"name": "x", "type": "int"}, {"name": "y"}]


def test_injector_spec_non_string_arg_ignored() -> None:
  """Verifies that unrecognized argument types in args sequence are skipped."""
  bad_args: list[Any] = ["x", 123, None]
  clean_args: list[Any] = StandardsInjector._serialize_args(None, args=bad_args)  # type: ignore[arg-type]
  assert clean_args == ["x"]


# --- Merged from test_injector_spec_missing.py ---


def test_injector_spec_missing() -> None:
  """Verifies the behavior of injector spec missing."""
  from ml_switcheroo.core.dsl import OperationDef, OpType, ParameterDef
  from ml_switcheroo.tools.injector_spec import StandardsInjector

  op_def: OperationDef = OperationDef(
    operation="Foo",
    description="Foo",
    variants={},
    op_type=OpType.CLASS,
    return_type="int",
    is_inplace=True,
    output_shape_calc="lambda x: x",
    std_args=[ParameterDef(name="a", type="int"), {"name": "b", "type": None}, {"name": "c", "type": "float"}, "d"],
  )
  injector: StandardsInjector = StandardsInjector(op_def)
  out: Dict[str, Any] = injector._serialize_op(op_def)
  assert out["op_type"] == "class"
  assert out["return_type"] == "int"
  assert out["is_inplace"] is True
  assert out["output_shape_calc"] == "lambda x: x"
  args: List[Any] = injector._serialize_args(getattr(op_def, "std_args"))
  assert args[1]["name"] == "b"
  assert "type" not in args[1]
  assert args[2]["name"] == "c"
  assert args[2]["type"] == "float"
  assert args[3] == "d"
  with __import__("unittest.mock").mock.patch("pathlib.Path.exists", return_value=True):
    with __import__("unittest.mock").mock.patch(
      "builtins.open", __import__("unittest.mock").mock.mock_open(read_data="bad json")
    ):
      assert injector.inject(dry_run=True) is True


def test_injector_spec_missing_more() -> None:
  """Verifies the behavior of injector spec missing more."""
  from ml_switcheroo.tools.injector_spec import StandardsInjector

  class DummyOpDef:
    """Docstring."""

    op_type: str = "function"

  injector: StandardsInjector = StandardsInjector(DummyOpDef())
  args: List[Any] = injector._serialize_args([{"name": "c", "foo": None}])
  assert args[0] == {"name": "c"}
  with __import__("unittest.mock").mock.patch("builtins.open", side_effect=OSError("fail")):
    import ml_switcheroo.tools.injector_spec

    ml_switcheroo.tools.injector_spec.Path = type("MockPath", (), {"exists": lambda self: True})
    pass


def test_injector_spec_write_parent_not_exist_extra() -> None:
  """Verifies the behavior of injector spec write parent not exist."""
  from ml_switcheroo.core.dsl import OperationDef
  from ml_switcheroo.tools.injector_spec import StandardsInjector

  op_def: OperationDef = OperationDef(operation="Foo", description="Foo", variants={})
  injector: StandardsInjector = StandardsInjector(op_def)

  class MockPath:
    """Docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      """Initializes the MockPath instance."""
      self.parent: Any = type(
        "MockParent", (), {"exists": lambda self: False, "mkdir": lambda self, parents=False, exist_ok=False: None}
      )()

    def exists(self) -> bool:
      """Mock implementation of exists."""
      return False

    def __truediv__(self, other: Any) -> Any:
      """Docstring."""
      return self

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.tools.injector_spec.resolve_semantics_dir", return_value=MockPath()
  ):
    with __import__("unittest.mock").mock.patch("builtins.open", __import__("unittest.mock").mock.mock_open()):
      injector.inject(dry_run=False)
