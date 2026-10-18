"""Test suite for the Autogen Ops module."""

import io
import typing

from typing import Any
from unittest import mock

import yaml

from ml_switcheroo.sphinx_ext.autogen_ops import (
  IndentedDumper,
  _build_yaml_entry,
  _write_index_file,
  _write_yaml_update,
  generate_op_docs,
)


class MockEnum:
  """Test helper class MockEnum."""

  def __init__(self, value: typing.Any) -> None:
    """Initializes the MockEnum instance.

    Args:
        value: Test argument or fixture.
    """
    self.value: typing.Any = value


def test_indented_dumper() -> None:
  """Verifies the behavior of indented dumper."""
  dumper: typing.Any = IndentedDumper(io.StringIO())
  result: typing.Any = dumper.increase_indent(flow=False, indentless=True)
  assert result is None


def test_build_yaml_entry() -> None:
  """Builds yaml entry."""
  definition: typing.Any = {
    "std_args": [
      "arg_str",
      ["arg_tuple_name", "arg_tuple_type"],
      {"name": "arg_dict_name", "type": "arg_dict_type", "optional": None},
      {"name": "arg_dict_2", "type": "arg_dict_type_2"},
    ],
    "op_type": MockEnum("test_enum_val"),
    "variants": {"jax": {"b": 2, "a": 1}, "torch": None, "numpy": {"c": 3}},
    "description": " Test desc `with` backticks ",
  }
  entry: typing.Any = _build_yaml_entry("test_op", definition)
  assert entry["operation"] == "test_op"
  assert entry["description"] == "Test desc `with` backticks"
  assert entry["op_type"] == "test_enum_val"
  assert len(entry["std_args"]) == 4
  assert entry["std_args"][0] == {"name": "arg_str", "type": "Any"}
  assert entry["std_args"][1] == {"name": "arg_tuple_name", "type": "arg_tuple_type"}
  assert entry["std_args"][2] == {"name": "arg_dict_name", "type": "arg_dict_type"}
  assert entry["std_args"][3] == {"name": "arg_dict_2", "type": "arg_dict_type_2"}
  assert "jax" in entry["variants"]
  assert list(entry["variants"]["jax"].keys()) == ["a", "b"]
  assert "torch" not in entry["variants"]
  assert "numpy" in entry["variants"]


def test_build_yaml_entry_minimal() -> None:
  """Builds yaml entry minimal."""
  entry: typing.Any = _build_yaml_entry("test_op", {})
  assert entry["operation"] == "test_op"
  assert entry["description"] == ""
  assert entry["op_type"] == "function"
  assert entry["std_args"] == []
  assert entry["variants"] == {}
  entry = _build_yaml_entry("test_op", {"std_args": [123]})
  assert entry["std_args"] == []
  entry = _build_yaml_entry("test_op", {"std_args": [["short"]]})
  assert entry["std_args"] == []


def test_write_yaml_update(tmp_path: typing.Any) -> None:
  """Verifies the behavior of write yaml update.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path: typing.Any = tmp_path / "operations.yaml"
  new_entries: typing.Any = [{"operation": "OpB", "val": 2}, {"operation": "OpA", "val": 1}]
  _write_yaml_update(out_path, new_entries)
  assert out_path.exists()
  content: typing.Any = out_path.read_text()
  assert "OpA" in content
  assert "OpB" in content
  new_entries_2: typing.Any = [{"operation": "OpA", "val": 99}, {"operation": "OpC", "val": 3}]
  _write_yaml_update(out_path, new_entries_2)
  loaded: typing.Any = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 3
  assert loaded[0]["operation"] == "OpA"
  assert loaded[0]["val"] == 99
  assert loaded[1]["operation"] == "OpB"
  assert loaded[2]["operation"] == "OpC"


def test_write_yaml_update_corrupt_existing(tmp_path: typing.Any) -> None:
  """Verifies the behavior of write yaml update corrupt existing.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path: typing.Any = tmp_path / "operations.yaml"
  out_path.write_text("invalid: yaml: [")
  new_entries: typing.Any = [{"operation": "OpA", "val": 1}]
  _write_yaml_update(out_path, new_entries)
  loaded: typing.Any = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 1
  assert loaded[0]["operation"] == "OpA"


def test_write_yaml_update_existing_not_list(tmp_path: typing.Any) -> None:
  """Verifies the behavior of write yaml update existing not list.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path: typing.Any = tmp_path / "operations.yaml"
  out_path.write_text("not_a_list: true")
  new_entries: typing.Any = [{"operation": "OpA", "val": 1}]
  _write_yaml_update(out_path, new_entries)
  loaded: typing.Any = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 1
  assert loaded[0]["operation"] == "OpA"


def test_write_yaml_update_ioerror(tmp_path: typing.Any) -> None:
  """Verifies the behavior of write yaml update ioerror.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path: typing.Any = tmp_path / "not_exist_dir" / "operations.yaml"
  new_entries: typing.Any = [{"operation": "OpA", "val": 1}]
  _write_yaml_update(out_path, new_entries)
  assert not out_path.exists()


def test_write_index_file(tmp_path: typing.Any) -> None:
  """Verifies the behavior of write index file.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_dir: typing.Any = tmp_path / "out"
  out_dir.mkdir()
  _write_index_file(out_dir, ["op1", "op2"])
  index_path: typing.Any = out_dir / "index.rst"
  assert index_path.exists()
  content: typing.Any = index_path.read_text()
  assert "Operation Reference" in content
  assert ".. toctree::" in content
  assert "   op1" in content
  assert "   op2" in content


class MockApp:
  """Test helper class MockApp."""

  def __init__(self, srcdir: typing.Any) -> None:
    """Initializes the MockApp instance.

    Args:
        srcdir: Test argument or fixture.
    """
    self.srcdir: typing.Any = srcdir


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs(
  mock_renderer_cls: typing.Any, mock_builder_cls: typing.Any, mock_manager_cls: typing.Any, tmp_path: typing.Any
) -> None:
  """Generates op documentation.

  Args:
      mock_renderer_cls (mock.MagicMock): Mock argument.
      mock_builder_cls (mock.MagicMock): Mock argument.
      mock_manager_cls (mock.MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_manager: typing.Any = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {
    "ValidOp": {"variants": {"jax": {}, "torch": {}}},
    "SkipMeOp": {"variants": {"jax": {}}},
    "validop": {"variants": {"jax": {}, "torch": {}}},
    "INDEX": {"variants": {"jax": {}, "torch": {}}},
    "Op/With!Special": {"variants": {"jax": {}, "torch": {}}},
  }
  mock_builder: typing.Any = mock.Mock()
  mock_builder_cls.return_value = mock_builder
  mock_builder.build.return_value = "mock_context"
  mock_renderer: typing.Any = mock.Mock()
  mock_renderer_cls.return_value = mock_renderer
  mock_renderer.render_rst.return_value = "mock_rst_content"
  srcdir: typing.Any = tmp_path / "docs"
  srcdir.mkdir()
  out_dir: typing.Any = srcdir / "ops"
  out_dir.mkdir()
  (out_dir / "stale.rst").write_text("stale")
  app: typing.Any = MockApp(str(srcdir))
  generate_op_docs(app)
  assert not (out_dir / "stale.rst").exists()
  assert (out_dir / "ValidOp.rst").exists()
  assert not (out_dir / "SkipMeOp.rst").exists()
  index_content: typing.Any = (out_dir / "index.rst").read_text()
  assert "   ValidOp" in index_content
  assert "   validop" not in index_content
  assert (out_dir / "index_op.rst").exists()
  assert (out_dir / "OpWithSpecial.rst").exists()
  assert (srcdir / "operations.yaml").exists()


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs_ioerror(
  mock_renderer_cls: typing.Any, mock_builder_cls: typing.Any, mock_manager_cls: typing.Any, tmp_path: typing.Any
) -> None:
  """Generates op documentation ioerror.

  Args:
      mock_renderer_cls (mock.MagicMock): Mock argument.
      mock_builder_cls (mock.MagicMock): Mock argument.
      mock_manager_cls (mock.MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_manager: typing.Any = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {"ValidOp": {"variants": {"jax": {}, "torch": {}}}}
  mock_builder: typing.Any = mock.Mock()
  mock_builder_cls.return_value = mock_builder
  mock_renderer: typing.Any = mock.Mock()
  mock_renderer_cls.return_value = mock_renderer
  srcdir: typing.Any = tmp_path / "docs"
  srcdir.mkdir()
  app: typing.Any = MockApp(str(srcdir))
  original_open: Any = open

  def mock_open(path: Any, *args: Any, **kwargs: Any) -> Any:
    """Provides a mock open for testing.

    Args:
        path (Any): Path to mock.
        *args (Any): Variable arguments.
        **kwargs (Any): Keyword arguments.

    Raises:
        IOError: Mock error.

    Returns:
        Any: Value returned by open.
    """
    if "ValidOp.rst" in str(path):
      raise IOError("mock error")
    return original_open(path, *args, **kwargs)

  with mock.patch("builtins.open", mock_open):
    generate_op_docs(app)
  out_dir: typing.Any = srcdir / "ops"
  assert not (out_dir / "ValidOp.rst").exists()


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs_empty(
  mock_renderer_cls: typing.Any, mock_builder_cls: typing.Any, mock_manager_cls: typing.Any, tmp_path: typing.Any
) -> None:
  """Generates op documentation empty.

  Args:
      mock_renderer_cls (mock.MagicMock): Mock argument.
      mock_builder_cls (mock.MagicMock): Mock argument.
      mock_manager_cls (mock.MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_manager: typing.Any = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {}
  srcdir: typing.Any = tmp_path / "docs"
  srcdir.mkdir()
  app: typing.Any = MockApp(str(srcdir))
  generate_op_docs(app)
  out_dir: typing.Any = srcdir / "ops"
  assert (out_dir / "index.rst").exists()


# --- Merged from test_autogen_ops_extra_loop.py ---


def test_autogen_ops_variant_with_enum() -> typing.Any:
  """Verify autogen ops variant with enum behavior.

  Returns:
      Computed test result or fixture value.
  """

  class DummyEnum:
    """Test helper class DummyEnum."""

    @property
    def value(self) -> typing.Any:
      """Execute value helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return "enum_val"

  definition: typing.Any = {"variants": {"jax": {"some_field": DummyEnum()}}}

  res: typing.Any = _build_yaml_entry("TestOp", definition)
  assert res["variants"]["jax"]["some_field"] == "enum_val"


def test_write_yaml_update_item_without_operation(tmp_path: typing.Any) -> None:
  """Test YAML update when existing file contains an item without 'operation' key.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path: typing.Any = tmp_path / "operations.yaml"
  out_path.write_text(yaml.dump([{"foo": "bar"}, {"operation": "Op1"}]))
  _write_yaml_update(out_path, [{"operation": "Op2"}])
  loaded: typing.Any = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 2
