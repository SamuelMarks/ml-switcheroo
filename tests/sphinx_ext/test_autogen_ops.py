import yaml
import shutil
from pathlib import Path
from unittest import mock

import pytest
from ml_switcheroo.sphinx_ext.autogen_ops import (
  IndentedDumper,
  _build_yaml_entry,
  _write_yaml_update,
  generate_op_docs,
  _write_index_file,
)


class MockEnum:
  def __init__(self, value):
    self.value = value


def test_indented_dumper():
  dumper = IndentedDumper(None)
  result = dumper.increase_indent(flow=False, indentless=True)
  assert result is None


def test_build_yaml_entry():
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

  assert len(entry["std_args"]) == 4
  assert entry["std_args"][0] == {"name": "arg_str", "type": "Any"}
  assert entry["std_args"][1] == {"name": "arg_tuple_name", "type": "arg_tuple_type"}
  assert entry["std_args"][2] == {"name": "arg_dict_name", "type": "arg_dict_type"}
  assert entry["std_args"][3] == {"name": "arg_dict_2", "type": "arg_dict_type_2"}

  assert "jax" in entry["variants"]
  assert list(entry["variants"]["jax"].keys()) == ["a", "b"]  # sorted
  assert "torch" not in entry["variants"]
  assert "numpy" in entry["variants"]


def test_build_yaml_entry_minimal():
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


def test_write_yaml_update(tmp_path):
  out_path = tmp_path / "operations.yaml"

  new_entries = [
    {"operation": "OpB", "val": 2},
    {"operation": "OpA", "val": 1},
  ]

  _write_yaml_update(out_path, new_entries)

  assert out_path.exists()
  content = out_path.read_text()
  assert "OpA" in content
  assert "OpB" in content

  new_entries_2 = [{"operation": "OpA", "val": 99}, {"operation": "OpC", "val": 3}]
  _write_yaml_update(out_path, new_entries_2)

  loaded = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 3
  assert loaded[0]["operation"] == "OpA"
  assert loaded[0]["val"] == 99
  assert loaded[1]["operation"] == "OpB"
  assert loaded[2]["operation"] == "OpC"


def test_write_yaml_update_corrupt_existing(tmp_path):
  out_path = tmp_path / "operations.yaml"
  out_path.write_text("invalid: yaml: [")

  new_entries = [{"operation": "OpA", "val": 1}]

  _write_yaml_update(out_path, new_entries)

  loaded = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 1
  assert loaded[0]["operation"] == "OpA"


def test_write_yaml_update_existing_not_list(tmp_path):
  out_path = tmp_path / "operations.yaml"
  out_path.write_text("not_a_list: true")

  new_entries = [{"operation": "OpA", "val": 1}]

  _write_yaml_update(out_path, new_entries)

  loaded = yaml.safe_load(out_path.read_text())
  assert len(loaded) == 1
  assert loaded[0]["operation"] == "OpA"


def test_write_yaml_update_ioerror(tmp_path):
  out_path = tmp_path / "not_exist_dir" / "operations.yaml"

  new_entries = [{"operation": "OpA", "val": 1}]

  _write_yaml_update(out_path, new_entries)
  assert not out_path.exists()


def test_write_index_file(tmp_path):
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
  def __init__(self, srcdir):
    self.srcdir = srcdir


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs(mock_renderer_cls, mock_builder_cls, mock_manager_cls, tmp_path):
  mock_manager = mock.Mock()
  mock_manager_cls.return_value = mock_manager

  mock_manager.get_known_apis.return_value = {
    "ValidOp": {"variants": {"jax": {}, "torch": {}}},
    "SkipMeOp": {"variants": {"jax": {}}},
    "validop": {"variants": {"jax": {}, "torch": {}}},  # collision with ValidOp
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

  # Check that SkipMeOp was skipped and not written
  # We use a completely unique name to avoid MacOS case-insensitivity tricking the assert
  assert not (out_dir / "SkipMeOp.rst").exists()

  # validop is skipped because of case-insensitive collision with ValidOp
  # Actually, on MacOS, validop.rst and ValidOp.rst are the same file, so we can't test existence.
  # But the file list generated shouldn't have duplicate or skipped entries.
  index_content = (out_dir / "index.rst").read_text()
  assert "   ValidOp" in index_content
  assert "   validop" not in index_content

  assert (out_dir / "index_op.rst").exists()
  assert (out_dir / "OpWithSpecial.rst").exists()

  assert (srcdir / "operations.yaml").exists()


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs_ioerror(mock_renderer_cls, mock_builder_cls, mock_manager_cls, tmp_path):
  mock_manager = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {
    "ValidOp": {"variants": {"jax": {}, "torch": {}}},
  }

  mock_builder = mock.Mock()
  mock_builder_cls.return_value = mock_builder

  mock_renderer = mock.Mock()
  mock_renderer_cls.return_value = mock_renderer

  srcdir = tmp_path / "docs"
  srcdir.mkdir()

  app = MockApp(str(srcdir))

  # Only raise IOError when writing the RST file inside the ops dir
  original_open = open

  def mock_open(path, *args, **kwargs):
    if "ValidOp.rst" in str(path):
      raise IOError("mock error")
    return original_open(path, *args, **kwargs)

  with mock.patch("builtins.open", mock_open):
    generate_op_docs(app)

  # generate_op_docs handles IOError gracefully for individual RST files
  out_dir = srcdir / "ops"
  assert not (out_dir / "ValidOp.rst").exists()


@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.DocContextBuilder")
@mock.patch("ml_switcheroo.sphinx_ext.autogen_ops.OpPageRenderer")
def test_generate_op_docs_empty(mock_renderer_cls, mock_builder_cls, mock_manager_cls, tmp_path):
  mock_manager = mock.Mock()
  mock_manager_cls.return_value = mock_manager
  mock_manager.get_known_apis.return_value = {}

  srcdir = tmp_path / "docs"
  srcdir.mkdir()

  app = MockApp(str(srcdir))

  generate_op_docs(app)

  out_dir = srcdir / "ops"
  assert (out_dir / "index.rst").exists()
