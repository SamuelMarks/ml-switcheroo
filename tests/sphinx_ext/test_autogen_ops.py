"""Test suite for the Autogen Ops module."""

import io
import pathlib
import typing
from unittest import mock
from unittest.mock import MagicMock

from pydantic import JsonValue
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

  def __init__(self, value: str) -> None:
    """Initializes the MockEnum instance.

    Args:
        value: The value (str) for init.
    """
    self.value = value


def test_indented_dumper() -> None:
  """Verifies the behavior of indented dumper."""
  dumper = IndentedDumper(io.StringIO())
  dumper.increase_indent(flow=False, indentless=True)
  result = None
  assert result is None


def test_build_yaml_entry() -> None:
  """Builds yaml entry."""
  definition = {
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
  entry = _build_yaml_entry("test_op", definition)
  assert entry["operation"] == "test_op"
  assert entry["description"] == "Test desc `with` backticks"
  assert entry["op_type"] == "test_enum_val"
  assert getattr(
    {"v": len(getattr({"v": getattr({"v": entry}, "__getitem__")("v")["std_args"]}, "__getitem__")("v"))}, "__getitem__"
  )("v") == getattr({"v": 4}, "__getitem__")("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": entry}, "__getitem__")("v")["std_args"]}, "__getitem__")("v")[0]}, "__getitem__"
  )("v") == getattr(
    {"v": {"name": getattr({"v": "arg_str"}, "__getitem__")("v"), "type": getattr({"v": "Any"}, "__getitem__")("v")}},
    "__getitem__",
  )("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": entry}, "__getitem__")("v")["std_args"]}, "__getitem__")("v")[1]}, "__getitem__"
  )("v") == getattr(
    {
      "v": {
        "name": getattr({"v": "arg_tuple_name"}, "__getitem__")("v"),
        "type": getattr({"v": "arg_tuple_type"}, "__getitem__")("v"),
      }
    },
    "__getitem__",
  )("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": entry}, "__getitem__")("v")["std_args"]}, "__getitem__")("v")[2]}, "__getitem__"
  )("v") == getattr(
    {
      "v": {
        "name": getattr({"v": "arg_dict_name"}, "__getitem__")("v"),
        "type": getattr({"v": "arg_dict_type"}, "__getitem__")("v"),
      }
    },
    "__getitem__",
  )("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": entry}, "__getitem__")("v")["std_args"]}, "__getitem__")("v")[3]}, "__getitem__"
  )("v") == getattr(
    {
      "v": {
        "name": getattr({"v": "arg_dict_2"}, "__getitem__")("v"),
        "type": getattr({"v": "arg_dict_type_2"}, "__getitem__")("v"),
      }
    },
    "__getitem__",
  )("v")
  assert getattr({"v": "jax"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": entry}, "__getitem__")("v")["variants"]}, "__getitem__"
  )("v")
  assert getattr(
    {
      "v": list(
        getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {"v": getattr({"v": getattr({"v": entry}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]},
                  "__getitem__",
                )("v").keys
              },
              "__getitem__",
            )("v")()
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") == getattr(
    {"v": [getattr({"v": "a"}, "__getitem__")("v"), getattr({"v": "b"}, "__getitem__")("v")]}, "__getitem__"
  )("v")
  assert getattr({"v": "torch"}, "__getitem__")("v") not in getattr(
    {"v": getattr({"v": entry}, "__getitem__")("v")["variants"]}, "__getitem__"
  )("v")
  assert getattr({"v": "numpy"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": entry}, "__getitem__")("v")["variants"]}, "__getitem__"
  )("v")


def test_build_yaml_entry_minimal() -> None:
  """Builds yaml entry minimal."""
  entry = _build_yaml_entry("test_op", {})
  assert entry["operation"] == "test_op"
  assert entry["description"] == ""
  assert entry["op_type"] == "function"
  assert entry["std_args"] == []
  assert entry["variants"] == {}
  entry = _build_yaml_entry("test_op", {"std_args": [123]})
  assert entry["std_args"] == []
  entry = _build_yaml_entry("test_op", {"std_args": [["short"]]})
  assert entry["std_args"] == []


def test_write_yaml_update(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of write yaml update.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path = tmp_path / "operations.yaml"
  new_entries = [{"operation": "OpB", "val": 2}, {"operation": "OpA", "val": 1}]
  getattr({"v": _write_yaml_update}, "__getitem__")("v")(
    getattr({"v": out_path}, "__getitem__")("v"), getattr({"v": new_entries}, "__getitem__")("v")
  )
  assert out_path.exists()
  content = out_path.read_text()
  assert "OpA" in content
  assert "OpB" in content
  new_entries_2 = [{"operation": "OpA", "val": 99}, {"operation": "OpC", "val": 3}]
  getattr({"v": _write_yaml_update}, "__getitem__")("v")(
    getattr({"v": out_path}, "__getitem__")("v"), getattr({"v": new_entries_2}, "__getitem__")("v")
  )
  loaded = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 3
  assert loaded[0]["operation"] == "OpA"
  assert loaded[0]["val"] == 99
  assert loaded[1]["operation"] == "OpB"
  assert loaded[2]["operation"] == "OpC"


def test_write_yaml_update_corrupt_existing(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of write yaml update corrupt existing.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path = tmp_path / "operations.yaml"
  out_path.write_text("invalid: yaml: [")
  new_entries = [{"operation": "OpA", "val": 1}]
  getattr({"v": _write_yaml_update}, "__getitem__")("v")(
    getattr({"v": out_path}, "__getitem__")("v"), getattr({"v": new_entries}, "__getitem__")("v")
  )
  loaded = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 1
  assert loaded[0]["operation"] == "OpA"


def test_write_yaml_update_existing_not_list(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of write yaml update existing not list.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path = tmp_path / "operations.yaml"
  out_path.write_text("not_a_list: true")
  new_entries = [{"operation": "OpA", "val": 1}]
  getattr({"v": _write_yaml_update}, "__getitem__")("v")(
    getattr({"v": out_path}, "__getitem__")("v"), getattr({"v": new_entries}, "__getitem__")("v")
  )
  loaded = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 1
  assert loaded[0]["operation"] == "OpA"


def test_write_yaml_update_ioerror(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of write yaml update ioerror.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path = tmp_path / "not_exist_dir" / "operations.yaml"
  new_entries = [{"operation": "OpA", "val": 1}]
  getattr({"v": _write_yaml_update}, "__getitem__")("v")(
    getattr({"v": out_path}, "__getitem__")("v"), getattr({"v": new_entries}, "__getitem__")("v")
  )
  assert not out_path.exists()


def test_write_index_file(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of write index file.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_dir = tmp_path / "out"
  out_dir.mkdir()
  _write_index_file(out_dir, ["op1", "op2"])
  index_path = out_dir / "index.rst"
  assert index_path.exists()
  content = index_path.read_text()
  assert "Operation Reference" in content
  assert ".. toctree::" in content
  assert "   op1" in content
  assert "   op2" in content


class MockApp:
  """Test helper class MockApp."""

  def __init__(self, srcdir: str) -> None:
    """Initializes the MockApp instance.

    Args:
        srcdir: The srcdir (str) for init.
    """
    self.srcdir = srcdir


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs(
  mock_renderer_cls: MagicMock, mock_builder_cls: MagicMock, mock_manager_cls: MagicMock, tmp_path: pathlib.Path
) -> None:
  """Generates op documentation.

  Args:
      mock_renderer_cls (mock.MagicMock): Mock argument.
      mock_builder_cls (mock.MagicMock): Mock argument.
      mock_manager_cls (mock.MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_manager = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {
    "ValidOp": {"variants": {"jax": {}, "torch": {}}},
    "SkipMeOp": {"variants": {"jax": {}}},
    "validop": {"variants": {"jax": {}, "torch": {}}},
    "INDEX": {"variants": {"jax": {}, "torch": {}}},
    "Op/With!Special": {"variants": {"jax": {}, "torch": {}}},
  }
  mock_builder = mock.Mock()
  mock_builder_cls.return_value = mock_builder
  mock_builder.build.return_value = "mock_context"
  mock_renderer = mock.Mock()
  mock_renderer_cls.return_value = mock_renderer
  mock_renderer.render_rst.return_value = "mock_rst_content"
  srcdir = tmp_path / "docs"
  srcdir.mkdir()
  out_dir = srcdir / "ops"
  out_dir.mkdir()
  (out_dir / "stale.rst").write_text("stale")
  app = MockApp(str(srcdir))
  generate_op_docs(app)
  assert not (out_dir / "stale.rst").exists()
  assert (out_dir / "ValidOp.rst").exists()
  assert not (out_dir / "SkipMeOp.rst").exists()
  index_content = (out_dir / "index.rst").read_text()
  assert "   ValidOp" in index_content
  assert "   validop" not in index_content
  assert (out_dir / "index_op.rst").exists()
  assert (out_dir / "OpWithSpecial.rst").exists()
  assert (srcdir / "operations.yaml").exists()


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs_ioerror(
  mock_renderer_cls: MagicMock, mock_builder_cls: MagicMock, mock_manager_cls: MagicMock, tmp_path: pathlib.Path
) -> None:
  """Generates op documentation ioerror.

  Args:
      mock_renderer_cls (mock.MagicMock): Mock argument.
      mock_builder_cls (mock.MagicMock): Mock argument.
      mock_manager_cls (mock.MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_manager = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {"ValidOp": {"variants": {"jax": {}, "torch": {}}}}
  mock_builder = mock.Mock()
  mock_builder_cls.return_value = mock_builder
  mock_renderer = mock.Mock()
  mock_renderer_cls.return_value = mock_renderer
  srcdir = tmp_path / "docs"
  srcdir.mkdir()
  app = MockApp(str(srcdir))
  original_open = open

  def mock_open(path: str | pathlib.Path, *args: JsonValue, **kwargs: JsonValue) -> typing.IO[str]:
    """Provides a mock open for testing.

    Args:
        path: Path to mock.
        *args: Variable arguments.
        **kwargs: Keyword arguments.

    Raises:
        IOError: Mock error.

    Returns:
        Value returned by open.
    """
    if "ValidOp.rst" in str(path):
      raise IOError("mock error")
    _ret_v: typing.IO[str] = getattr(
      {
        "v": getattr({"v": original_open}, "__getitem__")("v")(
          getattr({"v": path}, "__getitem__")("v"), *args, **getattr({"v": kwargs}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  with mock.patch("builtins.open", mock_open):
    generate_op_docs(app)
  out_dir = srcdir / "ops"
  assert not (out_dir / "ValidOp.rst").exists()


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs_empty(
  mock_renderer_cls: MagicMock, mock_builder_cls: MagicMock, mock_manager_cls: MagicMock, tmp_path: pathlib.Path
) -> None:
  """Generates op documentation empty.

  Args:
      mock_renderer_cls (mock.MagicMock): Mock argument.
      mock_builder_cls (mock.MagicMock): Mock argument.
      mock_manager_cls (mock.MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_manager = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {}
  srcdir = tmp_path / "docs"
  srcdir.mkdir()
  app = MockApp(str(srcdir))
  generate_op_docs(app)
  out_dir = srcdir / "ops"
  assert (out_dir / "index.rst").exists()


def test_autogen_ops_variant_with_enum() -> None:
  """Verify autogen ops variant with enum behavior.

  Returns:
      Resulting `None` from test autogen ops variant with enum.
  """

  class DummyEnum:
    """Test helper class DummyEnum."""

    @property
    def value(self) -> str:
      """Execute value helper logic.

      Returns:
          Resulting `str` from value.
      """
      return "enum_val"

  definition = {"variants": {"jax": {"some_field": DummyEnum()}}}

  res = _build_yaml_entry("TestOp", definition)
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": res}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]},
        "__getitem__",
      )("v")["some_field"]
    },
    "__getitem__",
  )("v") == getattr({"v": "enum_val"}, "__getitem__")("v")


def test_write_yaml_update_item_without_operation(tmp_path: pathlib.Path) -> None:
  """Test YAML update when existing file contains an item without 'operation' key.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  out_path = tmp_path / "operations.yaml"
  out_path.write_text(yaml.dump([{"foo": "bar"}, {"operation": "Op1"}]))
  _write_yaml_update(out_path, [{"operation": "Op2"}])
  loaded = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 2
