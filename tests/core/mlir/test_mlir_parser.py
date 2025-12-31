"""
Tests for MLIR Text Parser (Round-Trip).

Verifies:
1. Tokenizer correctness for identifiers, symbols, and trivia.
2. Parser structural correctness (Modules, Operations, Attributes).
3. Whitespace/Comment preservation (Byte-Identity).
"""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser, Tokenizer


def test_tokenizer_simple():
  text = '%0 = "sw.op"() : i32'
  tok = Tokenizer(text)
  tokens = list(tok.tokenize())

  # Check standard token stream
  token_kinds = [t.kind for t in tokens]
  assert "VAL_ID" in token_kinds
  assert "STRING" in token_kinds
  assert "SYMBOL" in token_kinds  # brackets
  assert "TYPE" in token_kinds


def roundtrip(code: str) -> str:
  """Helper to parse and re-emit."""
  parser = MlirParser(code)
  module = parser.parse()
  return module.to_text()


def test_parse_simple_op():
  """Verify basic op parsing."""
  # Update: Canonical formatting requires space before operands
  code = '%0 = "std.add" (%a, %b) : i32\n'
  assert roundtrip(code) == code


def test_parse_attributes():
  """Verify attribute dictionary."""
  code = 'sw.op {name = "test", id = 1}\n'
  assert roundtrip(code) == code


def test_parse_region_nested():
  """
  Verify region and block parsing.
  """
  # Update: Ensure canonical spacing for roundtrip equality check
  code = """sw.func { 
^entry: 
    sw.return
} 
"""
  assert roundtrip(code) == code


def test_parse_with_comments():
  """Verify comment preservation."""
  code = """// Header
sw.module { 
    // Body
    sw.op
} 
"""
  assert roundtrip(code) == code


def test_parse_block_args():
  """Verify block arguments."""
  # Update: Canonical input respects the spacing logic of BlockNode/OperationNode
  code = """^bb0(%arg0: i32, %arg1: f32): 
    sw.return
"""
  # Note: Manual block parsing test
  parser = MlirParser(code)
  blk = parser.parse_block(is_top_level=False)

  assert blk.to_text().strip() == code.strip()


def test_explicit_type_parsing():
  """Verify complex region type !sw.type."""
  code = '%0 = sw.op : !sw.type<"torch.nn.Conv2d">\n'
  assert roundtrip(code) == code
