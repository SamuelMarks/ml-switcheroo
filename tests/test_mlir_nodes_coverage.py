"""Module docstring."""

from ml_switcheroo.core.mlir.nodes import (
  OperationNode,
  StableHloConstantOp,
  ValueNode,
  AttributeNode,
  RegionNode,
  TypeNode,
  TriviaNode,
  ModuleNode,
  BlockNode,
)


def test_nodes_branches():
  """Docstring."""
  # ValueNode
  t = TypeNode(
    body="i32", leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  assert t.to_text() == "Li32T"

  v = ValueNode(
    name="%v", leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  assert v.to_text() == "L%vT"

  # AttributeNode
  a = AttributeNode(
    name="a",
    value='"1"',
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  assert a.to_text() == 'La = "1"T'

  a2 = AttributeNode(name="a", value='"1"', type_annotation="i32")
  assert a2.to_text() == 'a = "1" : i32'

  # OperationNode
  o1 = OperationNode(
    name='"sw.op"',
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T\n", kind="w")],
  )
  assert o1.to_text() == 'L"sw.op"T\n'

  o1b = OperationNode(name='"sw.op"', trailing_trivia=[TriviaNode(content="T", kind="w")])
  assert o1b.to_text() == '"sw.op"T\n'

  o2 = OperationNode(
    name='"sw.op"',
    results=[v],
    operands=[v],
    attributes=[a],
    regions=[RegionNode(blocks=[])],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  o2.to_text()

  o3 = OperationNode(name='"sw.op"', results=[v, v])
  o3.to_text()

  o4 = OperationNode(
    name='"sw.op"', name_trivia=[TriviaNode(content="N", kind="w")], operands=[v], attributes=[a], result_types=[t, t]
  )
  assert 'op"N' in o4.to_text()

  o5 = OperationNode(name='"sw.op"', result_types=[t])
  o5.to_text()

  # StableHloConstantOp
  so1 = StableHloConstantOp(
    name='"stablehlo.constant"',
    results=[v],
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  so1.to_text()

  so2 = StableHloConstantOp(
    name='"stablehlo.constant"',
    results=[v, v],
    attributes=[a, a],
    name_trivia=[TriviaNode(content="N", kind="w")],
    trailing_trivia=[TriviaNode(content="T\n", kind="w")],
  )
  so2.to_text()

  # BlockNode
  b1 = BlockNode(
    label="bb0",
    arguments=[(v, t)],
    operations=[o1],
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  b1.to_text()

  b2 = BlockNode(label="bb1", arguments=[(v, t), (v, t)], operations=[o1])
  b2.to_text()

  # RegionNode
  re1 = RegionNode(
    blocks=[b1], leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  re1.to_text()

  re2 = RegionNode(blocks=[b1, b2])
  re2.to_text()

  # ModuleNode
  m1 = ModuleNode(
    body=b1, leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  m1.to_text()

  m2 = ModuleNode(body=b1)
  m2.to_text()
