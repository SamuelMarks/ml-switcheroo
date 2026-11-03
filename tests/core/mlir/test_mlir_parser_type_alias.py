"""Tests for parsing MLIR type aliases."""

from tests.conftest import _test_getattr

from ml_switcheroo.core.mlir.parser import MlirParser


def test_parse_type_alias() -> None:
  """Test parsing a type alias definition."""
  code = "!my_type = i32"
  parser = MlirParser(code)
  module = parser.parse()
  alias = module.aliases[0]

  assert _test_getattr({"v": _test_getattr({"v": alias}, "__getitem__")("v").name}, "__getitem__")("v") == _test_getattr(
    {"v": "my_type"}, "__getitem__"
  )("v")
  assert _test_getattr(
    {"v": _test_getattr({"v": _test_getattr({"v": alias}, "__getitem__")("v").type_node}, "__getitem__")("v").body},
    "__getitem__",
  )("v") == _test_getattr({"v": "i32"}, "__getitem__")("v")
  assert alias.to_text() == "!my_type = i32"
