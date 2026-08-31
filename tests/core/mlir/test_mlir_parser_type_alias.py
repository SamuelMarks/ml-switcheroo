"""Tests for parsing MLIR type aliases."""

from ml_switcheroo.core.mlir.parser import MlirParser


def test_parse_type_alias() -> None:
  """Test parsing a type alias definition."""
  code = "!my_type = i32"
  parser = MlirParser(code)
  module = parser.parse()
  alias = module.aliases[0]

  assert alias.name == "my_type"
  assert alias.type_node.body == "i32"
  assert alias.to_text() == "!my_type = i32"
