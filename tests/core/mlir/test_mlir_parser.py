"""Test suite for the Mlir Parser module."""

from ml_switcheroo.core.mlir.parser import MlirParser


def test_parse_simple_op():
  """Parses simple op."""
  code = '%0 = "std.add"(%a, %b) : i32'
  parser = MlirParser(code)
  module = parser.parse()
  assert module is not None
  op = module.body.operations[0]
  assert op.name == '"std.add"'
  assert op.results[0].name == "%0"
  assert len(op.operands) == 2
  assert op.operands[0].name == "%a"
  assert op.operands[1].name == "%b"


def test_parse_attributes():
  """Parses attributes."""
  code = 'sw.op {name = "test", id = 1}'
  parser = MlirParser(code)
  module = parser.parse()
  op = module.body.operations[0]
  assert op.name == "sw.op"
  assert len(op.attributes) == 2
  assert op.attributes[0].name == "name"
  assert op.attributes[0].value == '"test"'


def test_parse_region_nested():
  """Parses region nested."""
  code = "sw.func {\n^entry:\n    sw.return\n}\n"
  parser = MlirParser(code)
  module = parser.parse()
  op = module.body.operations[0]
  assert op.name == "sw.func"
  assert len(op.regions) == 1
  assert op.regions[0].blocks[0].label == "^entry"
  assert op.regions[0].blocks[0].operations[0].name == "sw.return"


def test_parse_with_comments():
  """Parses with comments."""
  code = "// Header\nsw.module {\n    // Body\n    sw.op\n}\n"
  parser = MlirParser(code)
  module = parser.parse()
  op = module.body.operations[0]
  assert op.name == "sw.module"
  assert op.regions[0].blocks[0].operations[0].name == "sw.op"


def test_parse_block_args():
  """Parses block arguments."""
  code = "dummy {\n^bb0(%arg0: i32, %arg1: f32):\n    sw.return\n}"
  parser = MlirParser(code)
  module = parser.parse()
  blk = module.body.operations[0].regions[0].blocks[0]
  assert blk.label == "^bb0"
  assert len(blk.arguments) == 2
  assert blk.arguments[0][0].name == "%arg0"
  assert blk.arguments[0][1].body == "i32"


def test_explicit_type_parsing():
  """Verifies the behavior of explicit type parsing."""
  code = '%0 = sw.op : !sw.type<"torch.nn.Conv2d">\n'
  parser = MlirParser(code)
  module = parser.parse()
  op = module.body.operations[0]
  assert op.result_types[0].body == '!sw.type<"torch.nn.Conv2d">'


def test_parse_roundtrip_with_trivia():
  """Tests lossless roundtripping of MLIR code with complex trivia."""
  code = """
// Module comment
%0 = stablehlo.constant { value = 1 } : tensor<i32> // inline trailing
// Before block
sw.func {
  ^entry(%arg0 : i32):
    // Inside block
    %1 = "std.add"(%arg0, %0) { some_attr = "test" } : (i32, i32) -> i32
    sw.return %1
}
"""
  parser = MlirParser(code)
  module = parser.parse()
  out = module.to_text()

  # Verify comments and identifiers are preserved
  assert "// Module comment" in out
  assert "// inline trailing" in out
  assert "// Before block" in out
  assert "// Inside block" in out
  assert "%0" in out
  assert "stablehlo.constant" in out
  assert "value = 1" in out or "value = 1 " in out
  assert "tensor<i32>" in out
  assert "sw.func" in out
  assert "^entry" in out
