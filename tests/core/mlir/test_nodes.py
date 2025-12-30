"""
Tests for MLIR CST Object Model.

Verifies that:
1. Nodes can be instantiated.
2. Nodes render correct MLIR textual representation via `to_text`.
3. Trivia is preserved in output.
4. Nested structures (Regions/Blocks) render recursively.
"""

from ml_switcheroo.core.mlir.nodes import (
  TriviaNode,
  ValueNode,
  TypeNode,
  AttributeNode,
  OperationNode,
  BlockNode,
  RegionNode,
  ModuleNode,
)


def test_trivia_rendering():
  """Verify basic trivia preservation."""
  t1 = TriviaNode(content="\n", kind="newline")
  t2 = TriviaNode(content="// comment", kind="comment")
  assert t1.to_text() == "\n"
  assert t2.to_text() == "// comment"


def test_value_and_type_rendering():
  """Verify SSA values and Types."""
  v = ValueNode(name="%0")
  t = TypeNode(body="f32")
  assert v.to_text() == "%0"
  assert t.to_text() == "f32"


def test_attribute_rendering():
  """Verify attribute formatting."""
  # Simple
  a1 = AttributeNode(name="val", value="10")
  assert a1.to_text() == "val = 10"

  # Typed
  a2 = AttributeNode(name="metrics", value="dense<0>", type_annotation="tensor<1xi32>")
  assert a2.to_text() == "metrics = dense<0> : tensor<1xi32>"


def test_operation_simple():
  """
  Scenario: %0 = arith.addf(%a, %b) : f32
  """
  op = OperationNode(
    name="arith.addf",
    results=[ValueNode("%0")],
    operands=[ValueNode("%a"), ValueNode("%b")],
    result_types=[TypeNode("f32")],
  )
  # output contains implicit newline
  txt = op.to_text()
  assert txt.strip() == "%0 = arith.addf (%a, %b) : f32"
  assert txt.endswith("\n")


def test_operation_with_attributes_and_trivia():
  """
  Scenario:
      // Compute sum
      %sum = "sw.op"() { name = "add" } : () -> i32
  """
  op = OperationNode(
    name='"sw.op"',
    results=[ValueNode("%sum")],
    attributes=[AttributeNode(name="name", value='"add"')],
    leading_trivia=[
      TriviaNode("\n"),
      TriviaNode("// Compute sum\n"),
      TriviaNode("    "),  # indent
    ],
  )
  txt = op.to_text()

  expected_start = '\n// Compute sum\n    %sum = "sw.op" {name = "add"}'
  assert txt.startswith(expected_start)


def test_block_structure():
  """
  Scenario:
  ^bb0(%arg0: i32):
     %0 = op()
  """
  op = OperationNode(name="op", results=[ValueNode("%0")])

  blk = BlockNode(label="^bb0", arguments=[(ValueNode("%arg0"), TypeNode("i32"))], operations=[op])

  txt = blk.to_text()
  assert "^bb0(%arg0: i32):" in txt
  assert "%0 = op" in txt


def test_region_nesting():
  """
  Scenario:
  scf.if (%cond) {
     ^true:
       yield
  }
  """
  op_yield = OperationNode(name="yield")
  blk = BlockNode(label="^true", operations=[op_yield])
  region = RegionNode(blocks=[blk])

  op_if = OperationNode(name="scf.if", operands=[ValueNode("%cond")], regions=[region])

  txt = op_if.to_text()
  # Check structure
  assert "scf.if (%cond) {" in txt
  assert "^true:" in txt
  assert "yield" in txt
  assert "}" in txt


def test_module_node():
  """Verify top level container."""
  op = OperationNode(name="func.return")
  blk = BlockNode(label="", operations=[op])  # Implicit top block
  mod = ModuleNode(body=blk)

  assert "func.return" in mod.to_text()


def test_multiple_results_and_types():
  """
  Scenario: %0, %1 = op() : (i32, f32)
  """
  op = OperationNode(
    name="op", results=[ValueNode("%0"), ValueNode("%1")], result_types=[TypeNode("i32"), TypeNode("f32")]
  )
  txt = op.to_text()
  assert "%0, %1 = op" in txt
  assert ": (i32, f32)" in txt
