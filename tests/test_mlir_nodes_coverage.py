"""Module docstring."""

from ml_switcheroo.core.mlir.nodes import (
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  RegionNode,
  StableHloConstantOp,
  TriviaNode,
  TypeNode,
  ValueNode,
)


def test_nodes_branches() -> None:
  """Docstring."""
  # ValueNode
  t: TypeNode = TypeNode(
    body="i32", leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  assert t.to_text() == "Li32T"

  v: ValueNode = ValueNode(
    name="%v", leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  assert v.to_text() == "L%vT"

  # AttributeNode
  a: AttributeNode = AttributeNode(
    name="a",
    value='"1"',
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  assert a.to_text() == 'La = "1"T'

  a2: AttributeNode = AttributeNode(name="a", value='"1"', type_annotation="i32")
  assert a2.to_text() == 'a = "1" : i32'

  # OperationNode
  o1: OperationNode = OperationNode(
    name='"sw.op"',
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T\n", kind="w")],
  )
  assert o1.to_text() == 'L"sw.op"T\n'

  o1b: OperationNode = OperationNode(name='"sw.op"', trailing_trivia=[TriviaNode(content="T", kind="w")])
  assert o1b.to_text() == '"sw.op"T\n'

  o2: OperationNode = OperationNode(
    name='"sw.op"',
    results=[v],
    operands=[v],
    attributes=[a],
    regions=[RegionNode(blocks=[])],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  o2.to_text()

  o3: OperationNode = OperationNode(name='"sw.op"', results=[v, v])
  o3.to_text()

  o4: OperationNode = OperationNode(
    name='"sw.op"', name_trivia=[TriviaNode(content="N", kind="w")], operands=[v], attributes=[a], result_types=[t, t]
  )
  assert 'op"N' in o4.to_text()

  o5: OperationNode = OperationNode(name='"sw.op"', result_types=[t])
  o5.to_text()

  # StableHloConstantOp
  so1: StableHloConstantOp = StableHloConstantOp(
    name='"stablehlo.constant"',
    results=[v],
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  so1.to_text()

  so2: StableHloConstantOp = StableHloConstantOp(
    name='"stablehlo.constant"',
    results=[v, v],
    attributes=[a, a],
    name_trivia=[TriviaNode(content="N", kind="w")],
    trailing_trivia=[TriviaNode(content="T\n", kind="w")],
  )
  so2.to_text()

  # BlockNode
  b1: BlockNode = BlockNode(
    label="bb0",
    arguments=[(v, t)],
    operations=[o1],
    leading_trivia=[TriviaNode(content="L", kind="w")],
    trailing_trivia=[TriviaNode(content="T", kind="w")],
  )
  b1.to_text()

  b2: BlockNode = BlockNode(label="bb1", arguments=[(v, t), (v, t)], operations=[o1])
  b2.to_text()

  # RegionNode
  re1: RegionNode = RegionNode(
    blocks=[b1], leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  re1.to_text()

  re2: RegionNode = RegionNode(blocks=[b1, b2])
  re2.to_text()

  # ModuleNode
  m1: ModuleNode = ModuleNode(
    body=b1, leading_trivia=[TriviaNode(content="L", kind="w")], trailing_trivia=[TriviaNode(content="T", kind="w")]
  )
  m1.to_text()

  m2: ModuleNode = ModuleNode(body=b1)
  m2.to_text()


# --- Merged from test_mlir_nodes_coverage_more.py ---


def test_missing_nodes() -> None:
  """Docstring."""
  # 90 -> 92 (AttributeNode value is list)
  a_list: AttributeNode = AttributeNode(name="a", value=['"1"', '"2"'])
  assert a_list.to_text() == 'a = ["1", "2"]'

  # 126 -> 129 (BlockNode no arguments)
  # 129 -> 137 (BlockNode no label)
  b_nolabel: BlockNode = BlockNode(label="", operations=[OperationNode(name='"op"')])
  assert b_nolabel.to_text() == '"op"\n'

  # 133 -> 135 (BlockNode no ops, no trailing trivia, but has label)
  b_noops: BlockNode = BlockNode(label="bb0", arguments=[])
  assert b_noops.to_text() == "bb0:\n"

  # 223 -> 226 (OperationNode with regions, parts not empty, parts[-1] does not end with space)
  r: RegionNode = RegionNode(blocks=[])
  op_reg2: OperationNode = OperationNode(name='"sw.op"', regions=[r], attributes=[a_list])
  # attributes usually don't end with space, so let's see.
  assert " {" in op_reg2.to_text()

  # 267 -> 272 (StableHloConstantOp without results)
  so_nores: StableHloConstantOp = StableHloConstantOp(name='"stablehlo.constant"', attributes=[a_list])
  assert so_nores.to_text() == '"stablehlo.constant" [\'"1"\', \'"2"\']\n'

  # 280 -> 281 (StableHloConstantOp with name_trivia)
  so_nt: StableHloConstantOp = StableHloConstantOp(
    name='"stablehlo.constant"', attributes=[a_list], name_trivia=[TriviaNode(content=" N", kind="w")]
  )
  assert so_nt.to_text() == '"stablehlo.constant" N[\'"1"\', \'"2"\']\n'

  # 286 -> 287 (StableHloConstantOp without result_types)
  so_noval: StableHloConstantOp = StableHloConstantOp(name='"stablehlo.constant"')
  assert so_noval.to_text() == '"stablehlo.constant"\n'

  t: TypeNode = TypeNode(body="i32")
  # 288 -> 291
  so_types: StableHloConstantOp = StableHloConstantOp(name='"stablehlo.constant"', result_types=[t, t])
  assert so_types.to_text() == '"stablehlo.constant" : (i32, i32)\n'


def test_missing_nodes_2() -> None:
  """Docstring."""
  # 223 -> 226
  # parts[-1].endswith(" ")
  o: OperationNode = OperationNode(
    name='"sw.op"', regions=[RegionNode(blocks=[])], name_trivia=[TriviaNode(content=" ", kind="w")]
  )
  assert o.to_text() == '"sw.op" {}\n'

  # 288 -> 289
  t: TypeNode = TypeNode(body="i32")
  so: StableHloConstantOp = StableHloConstantOp(name='"stablehlo.constant"', result_types=[t])
  assert so.to_text() == '"stablehlo.constant" : i32\n'
