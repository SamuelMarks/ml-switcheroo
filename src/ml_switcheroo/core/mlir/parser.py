"""MLIR parser module.

This module implements a lexer and parser for MLIR text format,
producing a Concrete Syntax Tree (CST) using the pure-Python Lark parsing library.
"""

import re
from typing import List, Any, cast

from lark import Lark, Transformer, v_args
from lark.lexer import Lexer, Token

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  ModuleNode,
  BlockNode,
  RegionNode,
  OperationNode,
  ValueNode,
  TypeNode,
  AttributeNode,
  AttributeAliasDefNode,
)

TOKEN_REGEX = [
  ("COMMENT", r"//[^\n]*"),
  ("WS", r"[ \t\f\r\n]+"),
  ("CARET_ID", r"\^[a-zA-Z_0-9]+"),
  ("VAL_ID", r"%[a-zA-Z_0-9]+|%\d+"),
  ("SYM_ID", r"@[a-zA-Z_0-9]+"),
  ("ATTR_ALIAS_ID", r"#[a-zA-Z_0-9$._-]+"),
  ("TYPE", r"!sw\.type<[^>]+>|tensor<[^>]+>|![a-zA-Z_0-9\.<>]+|[iuf]\d+|index|none"),
  ("NUMBER", r"-?\d+(?:\.\d+)?"),
  ("STRING", r'"(?:[^"\\]|\\.)*"'),
  ("OPAQUE_DIALECT_CONTENTS", r"<[^>]+>"),
  ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_$.]*"),
  ("ARROW", r"->"),
  ("PUNCTUATION", r"[=,(){}\[\]:<>]"),
  ("MISMATCH", r"."),
]
tok_regex = "|".join("(?P<%s>%s)" % pair for pair in TOKEN_REGEX)


class MlirToken(Token):
  """Custom token that carries its leading trivia."""

  __slots__ = ("leading_trivia",)

  leading_trivia: List[Trivia]


class MlirLexer(Lexer):
  """Custom Lexer preserving trivia and matching MLIR tokens."""

  def __init__(self, lexer_conf: Any):
    """Init.

    Args:
        lexer_conf: Configuration for the lexer.
    """
    self.lexer_conf = lexer_conf

  def lex(self, data: str) -> Any:  # type: ignore[override]
    """Tokenize the input string and attach trivia.

    Args:
        data (str): The input string to lex.

    Yields:
        Any: The next matched token carrying its leading trivia.
    """
    leading: List[Trivia] = []
    for mo in re.finditer(tok_regex, data):
      kind = mo.lastgroup
      val = mo.group()
      if kind == "MISMATCH":
        raise ValueError(f"Unexpected '{val}'")
      if kind in ("WS", "COMMENT"):
        assert val is not None
        leading.append(Trivia(val))
      else:
        if kind == "PUNCTUATION":
          punct_map = {
            "=": "EQ",
            ",": "COMMA",
            "(": "LPAREN",
            ")": "RPAREN",
            "{": "LBRACE",
            "}": "RBRACE",
            "[": "LBRACK",
            "]": "RBRACK",
            ":": "COLON",
            "<": "LANGLE",
            ">": "RANGLE",
          }
          kind = punct_map[val]

        assert kind is not None
        assert val is not None
        t = MlirToken(kind, val)
        t.leading_trivia = list(leading)
        leading.clear()
        yield t


GRAMMAR = r"""
    ?start: module
    module: (operation | attribute_alias_def)*

    operation: [results EQ] op_name [SYM_ID] [operands] [dictionary_attribute] op_tail*

    attribute_alias_def: ATTR_ALIAS_ID EQ attr_value

    ?op_tail: regions | COLON result_types | ARROW result_types

    results: VAL_ID (COMMA VAL_ID)*
    op_name: IDENTIFIER | STRING | SYM_ID
    bare_id_list: IDENTIFIER (COMMA IDENTIFIER)*

    operands: LPAREN [operand (COMMA operand)*] RPAREN
            | operand (COMMA operand)*

    operand: VAL_ID [COLON TYPE]

    dictionary_attribute: LBRACE [attribute_entry (COMMA attribute_entry)*] RBRACE
    attribute_entry: attr_name EQ attr_value
    attr_name: IDENTIFIER | STRING
    attr_value: STRING | NUMBER | decimal_literal | TYPE | LBRACK [attr_value (COMMA attr_value)*] RBRACK | ATTR_ALIAS_ID | IDENTIFIER | dialect_attribute
    decimal_literal: NUMBER

    dialect_attribute: ATTR_ALIAS_ID (OPAQUE_DIALECT_CONTENTS | DOT IDENTIFIER)

    regions: region+
    region: LBRACE block* RBRACE

    block: [block_id [block_args] COLON] operation*
    block_args: LPAREN [block_arg (COMMA block_arg)*] RPAREN
    block_arg: VAL_ID COLON TYPE
    block_id: CARET_ID

    result_types: TYPE
                | LPAREN [TYPE (COMMA TYPE)*] RPAREN
                | LPAREN [TYPE (COMMA TYPE)*] RPAREN ARROW TYPE
                | LPAREN [TYPE (COMMA TYPE)*] RPAREN ARROW LPAREN [TYPE (COMMA TYPE)*] RPAREN

    // Missing grammar rules from LangRef
    dialect_type: "!" (opaque_dialect_type | pretty_dialect_type)
    dialect_type_body: LANGLE dialect_type_contents+ RANGLE
    dialect_type_contents: dialect_type_body | IDENTIFIER
    dictionary_properties: LANGLE dictionary_attribute RANGLE
    entry_block: operation+
    function_type: (TYPE | type_list_parens) ARROW (TYPE | type_list_parens)
    generic_operation: STRING [LPAREN value_use_list RPAREN] [successor_list]
    op_result: VAL_ID [COLON NUMBER]
    op_result_list: op_result (COMMA op_result)* EQ
    opaque_dialect_attribute: dialect_namespace dialect_attribute_body
    opaque_dialect_type: dialect_namespace dialect_type_body
    pretty_dialect_attribute: dialect_namespace DOT pretty_dialect_attribute_lead_ident
    pretty_dialect_attribute_lead_ident: IDENTIFIER
    pretty_dialect_type: dialect_namespace DOT pretty_dialect_type_lead_ident
    pretty_dialect_type_lead_ident: IDENTIFIER
    region_list: LPAREN region (COMMA region)* RPAREN
    ssa_use: value_use
    ssa_use_and_type: ssa_use COLON TYPE
    ssa_use_and_type_list: ssa_use_and_type (COMMA ssa_use_and_type)*
    successor: CARET_ID [COLON block_args]
    successor_list: LBRACK successor (COMMA successor)* RBRACK
    trailing_location: "loc" LPAREN STRING RPAREN
    type_alias: "!" IDENTIFIER
    type_alias_def: type_alias EQ TYPE
    type_list_no_parens: TYPE (COMMA TYPE)*
    type_list_parens: LPAREN [TYPE (COMMA TYPE)*] RPAREN
    value_id_and_type: VAL_ID COLON TYPE
    value_id_and_type_list: value_id_and_type (COMMA value_id_and_type)*
    value_id_list: VAL_ID (COMMA VAL_ID)*
    value_use: VAL_ID [ATTR_ALIAS_ID NUMBER]
    value_use_list: value_use (COMMA value_use)*
    dialect_namespace: IDENTIFIER
    dialect_attribute_body: LANGLE dialect_attribute_contents+ RANGLE
    dialect_attribute_contents: dialect_attribute_body | IDENTIFIER

    digit: /[0-9]/
    hex_digit: /[0-9a-fA-F]/
    hexadecimal_literal: /0x[0-9a-fA-F]+/
    id_punct: /[$._-]/
    letter: /[a-zA-Z]/
    suffix_id: /[0-9]+|([a-zA-Z$._-][a-zA-Z0-9$._-]*)/

    EQ: "="
    COMMA: ","
    LPAREN: "("
    RPAREN: ")"
    LBRACE: "{"
    RBRACE: "}"
    LBRACK: "["
    RBRACK: "]"
    COLON: ":"
    ARROW: "->"
    LANGLE: "<"
    RANGLE: ">"
    DOT: "."

    VAL_ID: /%./
    SYM_ID: /@./
    ATTR_ALIAS_ID: /#./
    TYPE: /!./
    NUMBER: /1/
    STRING: /"."/
    OPAQUE_DIALECT_CONTENTS: /<./
    IDENTIFIER: /a/
    CARET_ID: /\^./
"""


def _get_trivia(node: Any) -> List[Trivia]:
  """Extract leading trivia from a token or the first token in a tree.

  Args:
      node: The Lark AST Node or Token.

  Returns:
      A list of trivia items.
  """
  if hasattr(node, "leading_trivia"):
    res = node.leading_trivia
    node.leading_trivia = []
    return cast(List[Trivia], res)

  if hasattr(node, "children") and node.children:
    # Recursively find the first token with trivia
    return _get_trivia(node.children[0])

  return []


class MlirTransformer(Transformer[Any, Any]):
  """Transforms parsed AST nodes into MlirNode classes."""

  @v_args(inline=False)
  def module(self, children: List[Any]) -> ModuleNode:
    """Transform the top-level module rule.

    Args:
        children: Parsed children.

    Returns:
        The ModuleNode.
    """
    ops = [c for c in children if isinstance(c, OperationNode)]
    aliases = [c for c in children if isinstance(c, AttributeAliasDefNode)]
    leading = _get_trivia(children[0]) if children else []
    return ModuleNode(body=BlockNode(label="", operations=ops), aliases=aliases, leading_trivia=leading)

  @v_args(inline=False)
  def attribute_alias_def(self, children: List[Any]) -> "AttributeAliasDefNode":
    """Transform an attribute alias definition.

    Args:
        children: Parsed children.

    Returns:
        The AttributeAliasDefNode.
    """
    # ATTR_ALIAS_ID trivia? "=" trivia? attribute_value trivia?
    name_token = children[0]
    name = name_token.value

    # Find attribute_value
    val_node = next(c for c in children if getattr(c, "data", None) == "attr_value")
    if len(val_node.children) == 1:
      val = val_node.children[0]
      if hasattr(val, "value"):
        val_str = val.value
      else:
        val_str = str(val)
    else:
      # Array of values or something else
      val_str = str(val_node)

    leading = _get_trivia(children[0])

    # Handle trivia on trailing elements
    trailing = []
    if len(children) > 1 and getattr(children[-1], "data", None) == "trivia":
      trailing = _get_trivia(children[-1])

    # The "=" token is not captured, trivia after "=" is part of children if it matches trivia rule.
    # To keep it simple, we store raw string for now.

    return AttributeAliasDefNode(name=name, value_str=val_str, leading_trivia=leading, trailing_trivia=trailing)

  @v_args(inline=False)
  def operation(self, children: List[Any]) -> OperationNode:
    """Transform an operation rule into an OperationNode.

    Args:
        children: Parsed children.

    Returns:
        The constructed OperationNode.
    """
    op = OperationNode()
    for c in children:
      if c is not None:
        op.leading_trivia = _get_trivia(c)
        break

    i = 0
    while i < len(children):
      c = children[i]
      if isinstance(c, Token) and c.type == "EQ":
        pass
      elif getattr(c, "data", None) == "results":
        op.results = [ValueNode(name=v.value, leading_trivia=_get_trivia(v)) for v in c.children if v.type == "VAL_ID"]
      elif getattr(c, "data", None) == "op_name":
        op.name = c.children[0].value
      elif isinstance(c, Token) and c.type == "SYM_ID":
        # Usually `@main` after op_name, we can just append it to name or name_trivia
        triv = _get_trivia(c)
        op.name_trivia.extend(triv)
        op.name_trivia.append(Trivia(c.value))
      elif getattr(c, "data", None) == "operands":
        if c.children and isinstance(c.children[0], Token) and c.children[0].type == "LPAREN":
          op.has_parens = True
        else:
          op.has_parens = False
        for val in c.children:
          if getattr(val, "data", None) == "operand":
            v_tok = val.children[0]
            type_node = None
            colon_triv = []
            if len(val.children) > 2 and getattr(val.children[2], "type", None) == "TYPE":
              colon_triv = _get_trivia(val.children[1])
              type_node = TypeNode(body=val.children[2].value, leading_trivia=_get_trivia(val.children[2]))
            op.operands.append(
              ValueNode(name=v_tok.value, leading_trivia=_get_trivia(v_tok), type_node=type_node, colon_trivia=colon_triv)
            )
      elif isinstance(c, list):
        if len(c) > 0 and isinstance(c[0], AttributeNode):
          op.attributes = c
        elif len(c) > 0 and isinstance(c[0], RegionNode):
          op.regions.extend(c)
      elif getattr(c, "data", None) == "op_tail":
        op.op_tail_str = c.children[0].value
        op.op_tail_trivia = _get_trivia(c.children[0])
        for tail_child in c.children:
          if getattr(tail_child, "data", None) == "result_types":
            for t in tail_child.children:
              if getattr(t, "type", None) == "TYPE":
                op.result_types.append(TypeNode(body=t.value, leading_trivia=_get_trivia(t)))
      i += 1
    return op

  @v_args(inline=False)
  def dictionary_attribute(self, children: List[Any]) -> List[AttributeNode]:
    """Transform the dictionary_attribute rule into a list of AttributeNode.

    Args:
        children: Parsed children.

    Returns:
        List of parsed attributes.
    """
    attrs = []
    for c in children:
      if getattr(c, "data", None) == "attribute_entry":
        name = c.children[0].children[0].value
        val_node = c.children[2]

        if len(val_node.children) == 1:
          if hasattr(val_node.children[0], "value"):
            val = val_node.children[0].value
          else:
            val = "".join(t.value for t in val_node.children[0].scan_values(lambda v: isinstance(v, Token)))
        else:
          # Array of values
          val = [v.children[0].value for v in val_node.children if getattr(v, "data", None) == "attr_value"]

        attr = AttributeNode(name=name, value=val, leading_trivia=_get_trivia(c.children[0]))
        attrs.append(attr)
    return attrs

  @v_args(inline=False)
  def region(self, children: List[Any]) -> RegionNode:
    """Transform a region.

    Args:
        children: Parsed children.

    Returns:
        The constructed RegionNode.
    """
    leading = _get_trivia(children[0])
    trailing = _get_trivia(children[-1])
    blocks = [b for b in children if isinstance(b, BlockNode)]
    r = RegionNode(blocks=blocks)
    r.leading_trivia = leading
    r.trailing_trivia = trailing
    return r

  @v_args(inline=False)
  def regions(self, children: List[Any]) -> List[RegionNode]:
    """Transform the regions rule into a list of RegionNode.

    Args:
        children: Parsed children.

    Returns:
        List of parsed regions.
    """
    return [c for c in children if isinstance(c, RegionNode)]

  @v_args(inline=False)
  def block(self, children: List[Any]) -> BlockNode:
    """Transform the block rule into a BlockNode.

    Args:
        children: Parsed children.

    Returns:
        The constructed BlockNode.
    """
    label = ""
    args = []
    ops = []
    for c in children:
      if getattr(c, "data", None) == "block_id":
        label = c.children[0].value
      elif getattr(c, "data", None) == "block_args":
        for arg in c.children:
          if getattr(arg, "data", None) == "block_arg":
            v = ValueNode(name=arg.children[0].value, leading_trivia=_get_trivia(arg.children[0]))
            t = TypeNode(body=arg.children[2].value, leading_trivia=_get_trivia(arg.children[2]))
            args.append((v, t))
      elif isinstance(c, OperationNode):
        ops.append(c)

    leading = _get_trivia(children[0]) if children else []
    return BlockNode(label=label, arguments=args, operations=ops, leading_trivia=leading)


class MlirParser:
  """Parses a stream of MLIR tokens into a Concrete Syntax Tree."""

  def __init__(self, text: str):
    """Initialize the parser.

    Args:
        text (str): The MLIR source code to parse.
    """
    self.text = text
    self.parser = Lark(GRAMMAR, parser="earley", lexer=MlirLexer)
    self.transformer = MlirTransformer()

  def parse(self) -> ModuleNode:
    """Top-level parsing entry point.

    Returns:
        ModuleNode: The root of the MLIR CST.
    """
    if not self.text.strip():
      return ModuleNode()

    tree = self.parser.parse(self.text)
    node = self.transformer.transform(tree)
    return node  # type: ignore
