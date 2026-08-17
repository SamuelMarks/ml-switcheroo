"""Module docstring."""

from ml_switcheroo.core.mlir.nodes import (
  OperationNode,
  StableHloConstantOp,
  AttributeNode,
  RegionNode,
  TypeNode,
  TriviaNode,
  BlockNode,
)


def test_missing_nodes():
  """Docstring."""
  # 90 -> 92 (AttributeNode value is list)
  a_list = AttributeNode(name="a", value=['"1"', '"2"'])
  assert a_list.to_text() == 'a = ["1", "2"]'

  # 126 -> 129 (BlockNode no arguments)
  # 129 -> 137 (BlockNode no label)
  b_nolabel = BlockNode(label="", operations=[OperationNode(name='"op"')])
  assert b_nolabel.to_text() == '"op"\n'

  # 133 -> 135 (BlockNode no ops, no trailing trivia, but has label)
  b_noops = BlockNode(label="bb0", arguments=[])
  assert b_noops.to_text() == "bb0:\n"

  # 223 -> 226 (OperationNode with regions, parts not empty, parts[-1] does not end with space)
  r = RegionNode(blocks=[])
  op_reg2 = OperationNode(name='"sw.op"', regions=[r], attributes=[a_list])
  # attributes usually don't end with space, so let's see.
  assert " {" in op_reg2.to_text()

  # 267 -> 272 (StableHloConstantOp without results)
  so_nores = StableHloConstantOp(name='"stablehlo.constant"', attributes=[a_list])
  assert so_nores.to_text() == '"stablehlo.constant" [\'"1"\', \'"2"\']\n'

  # 280 -> 281 (StableHloConstantOp with name_trivia)
  so_nt = StableHloConstantOp(
    name='"stablehlo.constant"', attributes=[a_list], name_trivia=[TriviaNode(content=" N", kind="w")]
  )
  assert so_nt.to_text() == '"stablehlo.constant" N[\'"1"\', \'"2"\']\n'

  # 286 -> 287 (StableHloConstantOp without result_types)
  so_noval = StableHloConstantOp(name='"stablehlo.constant"')
  assert so_noval.to_text() == '"stablehlo.constant"\n'

  t = TypeNode(body="i32")
  # 288 -> 291
  so_types = StableHloConstantOp(name='"stablehlo.constant"', result_types=[t, t])
  assert so_types.to_text() == '"stablehlo.constant" : (i32, i32)\n'


def test_missing_nodes_2():
  """Docstring."""
  # 223 -> 226
  # parts[-1].endswith(" ")
  o = OperationNode(name='"sw.op"', regions=[RegionNode(blocks=[])], name_trivia=[TriviaNode(content=" ", kind="w")])
  assert o.to_text() == '"sw.op" {}\n'

  # 288 -> 289
  t = TypeNode(body="i32")
  so = StableHloConstantOp(name='"stablehlo.constant"', result_types=[t])
  assert so.to_text() == '"stablehlo.constant" : i32\n'
