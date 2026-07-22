"""Test suite for the Mlir Parser module."""

from ml_switcheroo.core.mlir.parser import MlirParser, Tokenizer


def test_tokenizer_simple():
  """Verifies the behavior of tokenizer simple."""
  text = '%0 = "sw.op"() : i32'
  tok = Tokenizer(text)
  tokens = list(tok.tokenize())
  token_kinds = [t.kind for t in tokens]
  assert "VAL_ID" in token_kinds
  assert "STRING" in token_kinds
  assert "SYMBOL" in token_kinds
  assert "TYPE" in token_kinds


def roundtrip(code: str) -> str:
  """Helper to roundtrip."""
  parser = MlirParser(code)
  module = parser.parse()
  return module.to_text()


def test_parse_simple_op():
  """Parses simple op."""
  code = '%0 = "std.add" (%a, %b) : i32\n'
  assert roundtrip(code) == code


def test_parse_attributes():
  """Parses attributes."""
  code = 'sw.op {name = "test", id = 1}\n'
  assert roundtrip(code) == code


def test_parse_region_nested():
  """Parses region nested."""
  code = "sw.func {\n^entry:\n    sw.return\n}\n"
  assert roundtrip(code) == code


def test_parse_with_comments():
  """Parses with comments."""
  code = "// Header\nsw.module {\n    // Body\n    sw.op\n}\n"
  assert roundtrip(code) == code


def test_parse_block_args():
  """Parses block arguments."""
  code = "^bb0(%arg0: i32, %arg1: f32):\n    sw.return\n"
  parser = MlirParser(code)
  blk = parser.parse_block(is_top_level=False)
  assert blk.to_text().strip() == code.strip()


def test_explicit_type_parsing():
  """Verifies the behavior of explicit type parsing."""
  code = '%0 = sw.op : !sw.type<"torch.nn.Conv2d">\n'
  assert roundtrip(code).strip() == '%0 = sw.op  : !sw.type<"torch.nn.Conv2d">'
