"""Test module."""

import ast
import typing
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter


def test_array_api_reader(tmp_path: Path) -> None:
  """Docstring."""
  # Create some dummy .py stubs
  (tmp_path / "valid.py").write_text('''
e = 2.718
"""Euler's number"""

PI: float = 3.14
"""Pi"""

def add(x: Array, y: Optional[Array]) -> Array:
    """Adds two arrays."""
    pass

def sub(x: Array, /, y: Array, *, out: Optional[Array] = None) -> Array:
    """Subtracts arrays."""
    pass

def _private(x):
    pass

_hidden = 1
''')

  (tmp_path / "_types.py").write_text("""
def should_skip():
    pass
""")

  (tmp_path / "__init__.py").write_text("""
def re_exported():
    pass
""")

  (tmp_path / "invalid.py").write_text("""
def broken(
""")

  importer = ArrayApiSpecImporter()

  # Empty dir
  empty_dir = tmp_path / "empty"
  empty_dir.mkdir()
  res_empty = importer.parse_folder(empty_dir)
  assert res_empty == {}

  # Valid dir
  res = importer.parse_folder(tmp_path)

  assert "add" in res
  assert res["add"]["description"] == "Adds two arrays."
  assert res["add"]["std_args"] == [("x", "Array"), ("y", "Optional[Array]")]

  assert "sub" in res
  assert res["sub"]["std_args"] == [("x", "Array"), ("y", "Array"), ("out", "Optional[Array]")]

  assert "e" in res
  assert res["e"]["description"] == "Euler's number"

  assert "PI" in res
  assert res["PI"]["description"] == "Pi"

  assert "re_exported" in res
  assert "_private" not in res
  assert "_hidden" not in res
  assert "should_skip" not in res


def test_parse_annotation() -> None:
  """Docstring."""
  importer = ArrayApiSpecImporter()

  # Test None
  assert importer._parse_annotation(None) == "Any"

  # Test Name
  node_name = ast.Name(id="int", ctx=ast.Load())
  assert importer._parse_annotation(node_name) == "int"

  # Test Constant
  node_const = ast.Constant(value="str")
  assert importer._parse_annotation(node_const) == "str"

  # Test Subscript
  node_sub = ast.Subscript(
    value=ast.Name(id="Optional", ctx=ast.Load()), slice=ast.Name(id="int", ctx=ast.Load()), ctx=ast.Load()
  )
  assert importer._parse_annotation(node_sub) == "Optional[int]"

  # Test Subscript with Tuple
  node_sub_tuple = ast.Subscript(
    value=ast.Name(id="Tuple", ctx=ast.Load()),
    slice=ast.Tuple(elts=[ast.Name(id="int", ctx=ast.Load()), ast.Name(id="float", ctx=ast.Load())], ctx=ast.Load()),
    ctx=ast.Load(),
  )
  assert importer._parse_annotation(node_sub_tuple) == "Tuple[int, float]"

  # Test BinOp (Union)
  node_binop = ast.BinOp(
    left=ast.Name(id="int", ctx=ast.Load()), op=ast.BitOr(), right=ast.Name(id="float", ctx=ast.Load())
  )
  assert importer._parse_annotation(node_binop) == "int | float"

  # Test BinOp fallback
  node_binop2 = ast.BinOp(
    left=ast.Name(id="int", ctx=ast.Load()), op=ast.Add(), right=ast.Name(id="float", ctx=ast.Load())
  )
  assert importer._parse_annotation(node_binop2) == "Any"

  # Test Attribute
  node_attr = ast.Attribute(value=ast.Name(id="types", ctx=ast.Load()), attr="NoneType", ctx=ast.Load())
  assert importer._parse_annotation(node_attr) == "types.NoneType"

  # Test Fallback
  node_fallback = ast.Pass()
  assert importer._parse_annotation(node_fallback) == "Any"


def test_get_assignment_name() -> None:
  """Docstring."""
  importer = ArrayApiSpecImporter()

  # Test Assign
  node_assign = ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=ast.Constant(value=1))
  assert importer._get_assignment_name(node_assign) == "x"

  node_assign2 = ast.Assign(targets=[ast.Attribute(value=ast.Name(id="self"), attr="x")], value=ast.Constant(value=1))
  assert importer._get_assignment_name(node_assign2) is None

  # Test AnnAssign
  node_annassign = ast.AnnAssign(
    target=ast.Name(id="y", ctx=ast.Store()), annotation=ast.Name(id="int"), value=ast.Constant(value=1), simple=1
  )
  assert importer._get_assignment_name(node_annassign) == "y"

  node_annassign2 = ast.AnnAssign(
    target=ast.Attribute(value=ast.Name(id="self"), attr="y"),
    annotation=ast.Name(id="int"),
    value=ast.Constant(value=1),
    simple=0,
  )
  assert importer._get_assignment_name(node_annassign2) is None

  # Test fallback
  node_fallback = ast.Pass()
  assert importer._get_assignment_name(node_fallback) is None


def test_clean_docstring() -> None:
  """Docstring."""
  importer = ArrayApiSpecImporter()
  assert importer._clean_docstring(None) == ""
  assert importer._clean_docstring("   ") == ""
  assert importer._clean_docstring("Line 1\nLine 2\n\nLine 3") == "Line 1 Line 2"


# --- Merged from test_array_api_reader_extra.py ---


def test_array_api_reader_relative_to_value_error(tmp_path: Path) -> None:
  """Docstring."""
  reader = ArrayApiSpecImporter()

  file_path: Path = tmp_path / "test.py"
  file_path.write_text("def foo(): pass")

  with patch("pathlib.Path.relative_to", side_effect=ValueError):
    res: dict[str, typing.Any] = reader._parse_stubs([file_path], tmp_path)

  assert "foo" in res
  assert res["foo"]["from"] == "test.py"


def test_array_api_reader_subscript_no_slice() -> None:
  """Docstring."""
  reader = ArrayApiSpecImporter()
  # Mock Subscript without slice
  node = ast.Subscript(value=ast.Name(id="List"), slice=ast.Name(id="Any"))
  if hasattr(node, "slice"):
    del node.slice
  assert reader._parse_annotation(node) == "List"


def test_array_api_reader_constants_branches(tmp_path: Path) -> None:
  """Test edge cases for constants docstring extraction."""
  (tmp_path / "consts.py").write_text("""
CONST_NO_DOC = 1
def not_doc(): pass

CONST_INT_EXPR = 2
42

CONST_LAST = 3
""")
  importer = ArrayApiSpecImporter()
  res = importer.parse_folder(tmp_path)
  assert res["CONST_NO_DOC"]["description"] == "Constant: CONST_NO_DOC"
  assert res["CONST_INT_EXPR"]["description"] == "Constant: CONST_INT_EXPR"
  assert res["CONST_LAST"]["description"] == "Constant: CONST_LAST"


def test_parse_snapshot_valid(tmp_path: Path) -> None:
  """Tests parsing a valid snapshot file with categories and operations."""
  snap_data = {
    "categories": {
      "elementwise": [
        {
          "api_path": "array_api.add",
          "name": "add",
          "docstring": "Calculates the sum.",
          "params": [
            {"name": "x1", "kind": "POSITIONAL_ONLY", "annotation": "Array"},
            {"name": "x2", "kind": "POSITIONAL_ONLY", "annotation": "Array"},
            {"name": "out", "kind": "KEYWORD_ONLY", "annotation": "Optional[Array]"},
            {"name": "extra", "kind": "VAR_POSITIONAL"},
          ],
          "returns_type": "Array",
        },
        "skip_non_dict",
        {"api_path": "array_api._private"},
      ]
    },
    "operations": {
      "abs": {
        "api_path": "array_api.abs",
        "name": "abs",
        "docstring": "Calculates absolute value.",
        "params": [
          {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "Array"},
          {"name": "", "kind": "POSITIONAL_OR_KEYWORD"},
          "skip_param",
        ],
      }
    },
  }
  import json

  snap_file = tmp_path / "array_api_v2024.12.json"
  snap_file.write_text(json.dumps(snap_data))

  importer = ArrayApiSpecImporter()
  res = importer.parse_snapshot(snap_file)
  assert "add" in res
  assert res["add"]["posonly_args"] == ["x1", "x2"]
  assert res["add"]["kwonly_args"] == ["out"]
  assert res["add"]["std_args"] == [("x1", "Array"), ("x2", "Array"), ("out", "Optional[Array]")]

  assert "abs" in res
  assert res["abs"]["std_args"] == [("x", "Array")]


def test_parse_snapshot_non_container_sections(tmp_path: Path) -> None:
  """Tests parsing snapshot when categories is not a dict or cat_items is not a list."""
  import json

  snap_file = tmp_path / "custom.json"
  snap_file.write_text(
    json.dumps(
      {
        "categories": "not_a_dict",
        "operations": "not_a_dict",
      }
    )
  )
  importer = ArrayApiSpecImporter()
  res = importer.parse_snapshot(snap_file)
  assert res == {}

  snap_file2 = tmp_path / "custom2.json"
  snap_file2.write_text(
    json.dumps(
      {
        "categories": {"cat1": "not_a_list"},
      }
    )
  )
  res2 = importer.parse_snapshot(snap_file2)
  assert res2 == {}


def test_parse_snapshot_errors(tmp_path: Path) -> None:
  """Tests error handling for missing and invalid snapshot files."""
  importer = ArrayApiSpecImporter()
  missing = tmp_path / "missing.json"

  import pytest

  with pytest.raises(FileNotFoundError):
    importer.parse_snapshot(missing)

  bad_json = tmp_path / "bad.json"
  bad_json.write_text("invalid json")
  with pytest.raises(ValueError, match="Corrupt snapshot JSON"):
    importer.parse_snapshot(bad_json)

  non_dict_json = tmp_path / "array.json"
  non_dict_json.write_text("[1, 2, 3]")
  with pytest.raises(ValueError, match="top-level JSON must be an object"):
    importer.parse_snapshot(non_dict_json)


def test_validate_function_signature() -> None:
  """Tests validation of function signatures against standard constraints."""
  importer = ArrayApiSpecImporter()
  import pytest

  # Valid
  assert importer.validate_function_signature("foo", {"std_args": [("x", "int"), ("y", "float")]})

  # Invalid op_name
  with pytest.raises(ValueError, match="Invalid operation name"):
    importer.validate_function_signature("", {"std_args": []})

  with pytest.raises(ValueError, match="Invalid operation name"):
    importer.validate_function_signature(123, {"std_args": []})  # type: ignore

  # Non-list std_args
  with pytest.raises(ValueError, match="std_args for bar must be a list"):
    importer.validate_function_signature("bar", {"std_args": "not a list"})

  # Malformed item in std_args
  with pytest.raises(ValueError, match="Malformed argument spec in baz"):
    importer.validate_function_signature("baz", {"std_args": ["not_a_tuple"]})

  # Invalid identifier
  with pytest.raises(ValueError, match="Invalid parameter name '123bad' in qux"):
    importer.validate_function_signature("qux", {"std_args": [("123bad", "int")]})


def test_sync_with_snapshot_and_golden(tmp_path: Path) -> None:
  """Tests sync_with_snapshot and golden regression against actual snapshot."""
  from ml_switcheroo.semantics.paths import resolve_snapshots_dir

  snap_path = resolve_snapshots_dir() / "array_api_v2024.12.json"
  importer = ArrayApiSpecImporter()

  if snap_path.exists():
    res = importer.sync_with_snapshot(snap_path)
    assert len(res) >= 100
    # Check core Array API canonical operations
    assert "add" in res
    assert "abs" in res
    assert "matmul" in res
    assert "sum" in res
    assert "concat" in res

    # Validate add signature: x1, x2
    add_args = [p[0] for p in res["add"]["std_args"]]
    assert "x1" in add_args and "x2" in add_args
  else:
    # Fallback to test sync_with_snapshot forwarding to parse_snapshot
    import json

    dummy = tmp_path / "dummy.json"
    dummy.write_text(json.dumps({"categories": {"ops": []}}))
    assert importer.sync_with_snapshot(dummy) == {}


def test_parse_snapshot_default_path(tmp_path: Path) -> None:
  """Tests parse_snapshot using default path resolution."""
  import json

  snap_file = tmp_path / "array_api_v2024.12.json"
  snap_file.write_text(json.dumps({"categories": {"array": []}}))
  importer = ArrayApiSpecImporter()

  with patch("ml_switcheroo.importers.array_api_reader.resolve_snapshots_dir", return_value=tmp_path):
    res = importer.parse_snapshot()
    assert res == {}
