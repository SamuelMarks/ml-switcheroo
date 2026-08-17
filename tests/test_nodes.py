"""Module docstring."""

from ml_switcheroo.core.mlir.nodes import (
  TriviaNode,
  ValueNode,
  TypeNode,
  AttributeNode,
  BlockNode,
  RegionNode,
  OperationNode,
  StableHloConstantOp,
  ModuleNode,
)


def test_trivia_node():
  """Docstring."""
  n = TriviaNode("  ")
  assert n.to_text() == "  "


def test_value_node():
  """Docstring."""
  n = ValueNode("%0")
  assert n.to_text() == "%0"

  n_trivia = ValueNode("%0", leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode("\n")])
  assert n_trivia.to_text() == " %0\n"


def test_type_node():
  """Docstring."""
  n = TypeNode("!sw.type")
  assert n.to_text() == "!sw.type"

  n_trivia = TypeNode("i32", leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode("\n")])
  assert n_trivia.to_text() == " i32\n"


def test_attribute_node():
  """Docstring."""
  # String
  n_str = AttributeNode(name="sym_name", value='"my_func"')
  assert n_str.to_text() == 'sym_name = "my_func"'

  # List
  n_list = AttributeNode(name="bases", value=['"Base1"', '"Base2"'])
  assert n_list.to_text() == 'bases = ["Base1", "Base2"]'

  # Type Annotation
  n_type = AttributeNode(name="value", value="42", type_annotation="i32")
  assert n_type.to_text() == "value = 42 : i32"

  # Trivia
  n_trivia = AttributeNode(
    name="name", value='"val"', leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode(" ")]
  )
  assert n_trivia.to_text() == ' name = "val" '


def test_block_node():
  """Docstring."""
  # Empty
  n_empty = BlockNode(label="^bb0")
  assert n_empty.to_text() == "^bb0:\n"

  # With args
  arg_val = ValueNode("%arg0")
  arg_typ = TypeNode("i32")
  n_args = BlockNode(label="^bb0", arguments=[(arg_val, arg_typ)])
  assert n_args.to_text() == "^bb0(%arg0: i32):\n"

  # With ops
  op = OperationNode(name="sw.return")
  n_ops = BlockNode(label="", operations=[op])
  assert n_ops.to_text() == "sw.return\n"  # newline added by operation

  # Trivia
  n_trivia = BlockNode(label="^bb0", leading_trivia=[TriviaNode("\n")])
  assert n_trivia.to_text() == "\n^bb0:\n"


def test_region_node():
  """Docstring."""
  b = BlockNode(label="^bb0")
  n = RegionNode(blocks=[b])
  assert n.to_text() == "{^bb0:\n}"

  n_trivia = RegionNode(blocks=[], leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode(" ")])
  assert n_trivia.to_text() == " {} "


def test_operation_node():
  """Docstring."""
  # Bare
  n = OperationNode(name="sw.return")
  assert n.to_text() == "sw.return\n"

  # Results + Operands
  n_res_op = OperationNode(name="sw.add", results=[ValueNode("%res")], operands=[ValueNode("%a"), ValueNode("%b")])
  assert n_res_op.to_text() == "%res = sw.add (%a, %b)\n"

  # Attributes + Result Types
  n_attr_type = OperationNode(
    name="sw.constant", attributes=[AttributeNode(name="value", value="42")], result_types=[TypeNode("i32")]
  )
  assert n_attr_type.to_text() == "sw.constant {value = 42} : i32\n"

  # Multi result types
  n_multi_type = OperationNode(name="sw.div", result_types=[TypeNode("i32"), TypeNode("i32")])
  assert n_multi_type.to_text() == "sw.div : (i32, i32)\n"

  # Regions
  r = RegionNode(blocks=[])
  n_region = OperationNode(name="sw.func", regions=[r])
  assert n_region.to_text() == "sw.func {}\n"

  # Trivia
  n_trivia = OperationNode(
    name="sw.test",
    leading_trivia=[TriviaNode("// l")],
    name_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode("// t\n")],
  )
  assert n_trivia.to_text() == "// lsw.test // t\n"


def test_stablehlo_constant_op():
  """Docstring."""
  n = StableHloConstantOp(
    name="stablehlo.constant",
    results=[ValueNode("%0")],
    attributes=[AttributeNode(name="value", value="dense<1.0>")],
    result_types=[TypeNode("tensor<f32>")],
  )
  assert n.to_text() == "%0 = stablehlo.constant dense<1.0> : tensor<f32>\n"

  # Multi types
  n_multi = StableHloConstantOp(
    name="stablehlo.constant",
    attributes=[AttributeNode(name="value", value="dense<1>")],
    result_types=[TypeNode("i32"), TypeNode("i64")],
  )
  assert n_multi.to_text() == "stablehlo.constant dense<1> : (i32, i64)\n"

  # Name Trivia
  n_trivia = StableHloConstantOp(
    name="stablehlo.constant",
    name_trivia=[TriviaNode(" ")],
    attributes=[AttributeNode(name="value", value="1")],
    trailing_trivia=[TriviaNode(" // a\n")],
  )
  assert n_trivia.to_text() == "stablehlo.constant 1 // a\n"


def test_module_node():
  """Docstring."""
  b = BlockNode(label="^bb0")
  m = ModuleNode(body=b)
  assert m.to_text() == "^bb0:\n"

  m_trivia = ModuleNode(body=b, leading_trivia=[TriviaNode("// top\n")], trailing_trivia=[TriviaNode("// bot\n")])
  assert m_trivia.to_text() == "// top\n^bb0:\n// bot\n"
