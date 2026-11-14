"""Unit tests for `MlirLexer`, `MlirParser`, and `MlirTransformer` (`ml_switcheroo.core.mlir.parser`)."""

import pytest
from lark import Token, Tree

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  AttributeAliasDefNode,
  AttributeNode,
  ModuleNode,
  OperationNode,
  RegionNode,
  TypeAliasDefNode,
)
from ml_switcheroo.core.mlir.parser import MlirLexer, MlirParser, MlirToken, MlirTransformer, _get_trivia

_NL = chr(10)


def test_parse_basic_roundtrip_and_invalid_tokens() -> None:
  """Verify empty modules, roundtrip formatting with trivia, and `ValueError` on invalid tokens."""
  assert MlirParser("").parse().body.operations == []
  assert MlirParser("   ").parse().body.operations == []

  with pytest.raises(ValueError, match="Unexpected"):
    MlirParser("~").parse()

  from ml_switcheroo.exceptions import SyntaxParseError

  with pytest.raises(SyntaxParseError, match="Failed to parse MLIR"):
    MlirParser("%0 = ").parse()

  basic = _NL.join(
    [
      "module {",
      "      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {",
      "        %0 = stablehlo.add %arg0, %arg0 : tensor<f32>",
      "        return %0 : tensor<f32>",
      "      }",
      "    }",
    ]
  )
  mod = MlirParser(basic).parse()
  assert isinstance(mod, ModuleNode)
  assert mod.to_text() == basic

  trivia_code = _NL.join(
    [
      "// Module comment",
      "%0 = stablehlo.constant { value = 1 } : tensor<i32> // inline trailing",
      "// Before block",
      "sw.func {",
      "  ^entry(%arg0 : i32):",
      "    // Inside block",
      '    %1 = "std.add"(%arg0, %0) { some_attr = "test" } : (i32, i32) -> i32',
      "    sw.return %1",
      "}",
    ]
  )
  out = MlirParser(trivia_code).parse().to_text()
  assert "// Module comment" in out
  assert "// inline trailing" in out
  assert "// Before block" in out
  assert "// Inside block" in out
  assert "^entry" in out


def test_parse_custom_and_generic_operations() -> None:
  """Verify custom ops, generic ops, typed/indexed operands, successors, properties, regions, and locations."""
  custom = MlirParser('%res1, %res2 = "dialect.op1" @symbol (%arg0 : !type1, %arg1 : !type2)').parse()
  op0 = custom.body.operations[0]
  assert op0.name == '"dialect.op1"'
  assert [r.name for r in op0.results] == ["%res1", "%res2"]
  assert op0.operands[0].type_node is not None
  assert op0.operands[0].type_node.body == "!type1"
  assert op0.operands[1].type_node is not None
  assert op0.operands[1].type_node.body == "!type2"

  sym_fn = MlirParser("func.func @main() { return }").parse()
  fn_op = sym_fn.body.operations[0]
  assert fn_op.name == "func.func"
  assert isinstance(fn_op.name_trivia, list)
  assert fn_op.name_trivia[-1].text == "@main"

  indexed = MlirParser("sw.op(%0#1, @symbol : !type)").parse().body.operations[0]
  assert indexed.operands[0].use_index == 1
  assert indexed.operands[1].name == "@symbol"

  generic_src = _NL.join(
    [
      '%0:1, %1:1 = "generic.test"()',
      '%2:1 = "generic.add"(%a, %b)',
      '%3:1 = "generic.br"()[^bb1, ^bb2]',
      '%4:1 = "generic.op"() <{ prop = "val" }>',
      '%5:1 = "generic.multi"(%a, %b) : (i32, f32) -> (i32, f32)',
      '%6:1 = "generic.single"(%a) : i32 -> i32',
      '%7:1 = "generic.mix"(%a) : i32 -> (i32, f32)',
      '%8:1 = "generic.mix2"(%a, %b) : (i32, f32) -> i32',
      '%9:1 = "generic.extra_types"() : (i32, f32) -> i32',
      '%10:1 = "generic.extra_types2"() : i32 -> i32',
      '"my.op"() ( { } )',
      'sw.op() loc("my_file.py:10:2")',
      '"my.op"() -> i32',
      '"my.op"() : () -> ()',
    ]
  )
  g_ops = MlirParser(generic_src).parse().body.operations
  assert g_ops[0].is_generic is True
  assert g_ops[0].name == "generic.test"
  assert [v.name for v in g_ops[1].operands] == ["%a", "%b"]
  assert g_ops[2].successors == ["^bb1", "^bb2"]
  assert g_ops[3].properties[0].name == "prop"
  assert len(g_ops[4].result_types) == 2
  assert len(g_ops[5].result_types) == 1
  assert g_ops[11].location == '"my_file.py:10:2"'


def test_parse_attributes_blocks_and_aliases() -> None:
  """Verify attribute syntax variants, block arguments, attribute aliases, and type aliases."""
  attrs_src = _NL.join(
    [
      '#map = "my_string"',
      "#map2 = [1, 2]",
      "#map3 = f32",
      "!my_type = !sw.type<f32>",
      'sw.op {name = "test", arr = [1, 2], align = none, items = [a, b], ref = #map, d = #dialect.attr, opq = #dialect<data>}',
      '%0 = sw.op : !sw.type<"torch.nn.Conv2d">',
      "dummy {",
      "^bb0(%arg0: i32, %arg1: f32):",
      "    sw.return",
      "}",
    ]
  )
  mod = MlirParser(attrs_src).parse()
  assert len(mod.aliases) == 4
  a0, a1, a2, a3 = mod.aliases
  assert isinstance(a0, AttributeAliasDefNode)
  assert a0.name == "#map"
  assert a0.value_str == '"my_string"'
  assert isinstance(a1, AttributeAliasDefNode)
  assert a1.name == "#map2"
  assert isinstance(a2, AttributeAliasDefNode)
  assert a2.name == "#map3"
  assert a2.value_str == "f32"
  assert isinstance(a3, TypeAliasDefNode)
  assert a3.name == "my_type"
  assert a3.type_node is not None
  assert a3.type_node.body == "!sw.type<f32>"

  attr_op = mod.body.operations[0]
  assert attr_op.attributes[1].value == ["1", "2"]
  assert attr_op.attributes[2].value == "none"
  assert attr_op.attributes[3].value == ["a", "b"]
  assert attr_op.attributes[4].value == "#map"
  assert attr_op.attributes[5].value == "#dialect.attr"
  assert attr_op.attributes[6].value == "#dialect<data>"

  blk = mod.body.operations[2].regions[0].blocks[0]
  assert blk.label == "^bb0"
  assert len(blk.arguments) == 2


def test_lexer_and_transformer_direct_edge_cases() -> None:
  """Verify `MlirLexer`, `_get_trivia`, and `MlirTransformer` direct AST construction branches."""

  class _LexerStateStub:
    """Lexer state stub exposing `text`."""

    def __init__(self, text: str) -> None:
      """Initialize `text` field.

      Args:
          text: Raw MLIR source text.
      """
      self.text = text

  tokens = list(MlirLexer(None).lex(_LexerStateStub('%0 = "dummy"()')))
  assert tokens[0].value == "%0"

  assert _get_trivia(None) == []
  assert _get_trivia(Tree[Token]("empty", [])) == []

  t = MlirTransformer()
  assert t.module([]).leading_trivia == []
  assert t.block([]).leading_trivia == []
  assert len(t.block(["ignored_string", OperationNode(name="op")]).operations) == 1

  reg = RegionNode()
  assert t.regions([reg, "dummy"]) == [reg]

  attr = AttributeNode(name="attr")
  assert t.operation([None]).leading_trivia == []
  assert t.operation([[attr]]).attributes == [attr]
  assert t.operation([[reg]]).regions == [reg]
  assert t.operation([[]]).attributes == []
  assert t.operation([["other"]]).attributes == []
  assert t.operation([Tree[Token]("operands", [])]).has_parens is False
  assert t.operation([Tree[Token]("op_tail", [])]).op_tail_str == ""
  assert t.operation([Tree("function_type", [Token("TYPE", "i32")])]).result_types == []
  assert (
    t.operation([Tree("function_type", [Token("OTHER", "x"), Token("ARROW", "->"), Token("OTHER", "y")])]).result_types
    == []
  )

  # Direct transformer trees for `op_result_list`, `successor_list`, `region_list`, `operands`, `function_type`, `op_tail`
  def _tree(data: str, children: list[Token | Tree[Token]]) -> Tree[Token]:
    """Construct a `Tree[Token]` with explicit `Token | Tree[Token]` child list typing.

    Args:
        data: Lark grammar rule name.
        children: Child tokens or nested `Tree[Token]` nodes.

    Returns:
        Typed `Tree[Token]` instance.
    """
    return Tree(data, children)

  op_res = _tree(
    "op_result_list",
    [
      _tree("op_result", [Token("VAL_ID", "%0")]),
      _tree("op_result", [_tree("unknown", [])]),
      _tree("unknown_node", []),
    ],
  )
  succ_ok = _tree("successor_list", [_tree("successor", [_tree("caret_id", [Token("CARET", "^"), Token("ID", "bb3")])])])
  succ_skip = _tree("successor_list", [_tree("successor", [Token("OTHER", "dummy")])])
  reg_list = _tree("region_list", [Token("LPAREN", "("), _tree("region", [Token("LBRACE", "{")])])
  op_args = _tree("operands", [_tree("operand_list", [_tree("operand", [_tree("value_use", [Token("VAL_ID", "%0")])])])])
  op_args_direct = _tree("operands", [_tree("operand_list", [_tree("operand", [Token("VAL_ID", "%0")])])])
  fn_parens = _tree(
    "function_type",
    [
      _tree("unknown_data", []),
      _tree("type_list_parens", [Token("OTHER", "dummy"), Token("TYPE", "i32")]),
      Token("ARROW", "->"),
      _tree("unknown_data", []),
      _tree("type_list_parens", [Token("OTHER", "dummy"), Token("TYPE", "f32")]),
    ],
  )
  tail_colon = _tree("op_tail", [Token("COLON", ":"), _tree("result_types", [Token("TYPE", "i64")])])
  built_op = t.operation([op_res, succ_ok, reg_list, op_args, fn_parens, tail_colon])
  assert built_op.successors == ["^bb3"]
  assert len(built_op.regions) >= 1
  assert (
    t.operation([op_args_direct, _tree("op_tail", []), _tree("function_type", [Token("TYPE", "i32")]), succ_skip])
    .operands[0]
    .name
    == "%0"
  )
  assert t.operation([]).name == ""

  # `type_alias_def` and `attribute_alias_def` with trailing trivia and custom tree child
  tok_t = MlirToken("TYPE", "!t")
  tok_t.leading_trivia = []
  eq_tok = MlirToken("EQ", "=")
  eq_tok.leading_trivia = []
  body_tok = MlirToken("TYPE", "!other")
  body_tok.leading_trivia = []
  triv_tok = MlirToken("WS", " ")
  triv_tok.leading_trivia = [Trivia("// comment")]
  triv_tree: Tree[MlirToken] = Tree("trivia", [triv_tok])

  t_alias = t.type_alias_def([Tree("type_alias", [tok_t]), eq_tok, body_tok, triv_tree])
  assert t_alias.name == "t"
  assert len(t_alias.trailing_trivia) == 1

  name_tok = MlirToken("ATTR_ALIAS_ID", "#foo")
  name_tok.leading_trivia = []
  triv_tok2 = MlirToken("WS", " ")
  triv_tok2.leading_trivia = [Trivia("// comment")]
  triv_tree2: Tree[MlirToken] = Tree("trivia", [triv_tok2])
  val_tree = _tree("attribute_value", [_tree("custom_val", [])])
  a_alias = t.attribute_alias_def([Tree("attribute_alias", [name_tok]), eq_tok, val_tree, triv_tree2])
  assert a_alias.name == "#foo"
  assert "Tree" in a_alias.value_str
  assert a_alias.trailing_trivia[0].text == "// comment"

  from ml_switcheroo.exceptions import SyntaxParseError

  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(MlirTransformer, "transform", lambda self, tree: "not_a_module")
    with pytest.raises(SyntaxParseError, match="Expected ModuleNode root"):
      MlirParser("sw.op").parse()
