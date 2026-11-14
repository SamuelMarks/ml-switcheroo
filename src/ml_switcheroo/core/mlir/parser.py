"""MLIR parser module.

This module implements a lexer and parser for MLIR text format,
producing a Concrete Syntax Tree (CST) using the pure-Python Lark parsing library.
"""

import re
import typing

from lark import Lark, Transformer, Tree, v_args
from lark.exceptions import LarkError
from lark.lexer import Lexer, Token

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  AliasNodeUnion,
  AttributeAliasDefNode,
  AttributeNode,
  BlockNode,
  MlirNode,
  ModuleNode,
  OperationNode,
  RegionNode,
  TypeAliasDefNode,
  TypeNode,
  ValueNode,
)
from ml_switcheroo.exceptions import SyntaxParseError

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
  ("OPAQUE_DIALECT_CONTENTS", r"<(?![{])[^>]+>"),
  ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_$.]*"),
  ("ARROW", r"->"),
  ("PUNCTUATION", r"[=,(){}\[\]:<>.]"),
  ("MISMATCH", r"."),
]
tok_regex = "|".join("(?P<%s>%s)" % pair for pair in TOKEN_REGEX)


@typing.runtime_checkable
class _LexerTextProtocol(typing.Protocol):
  """Protocol for Lark LexerState exposing `text`.

  Attributes:
      text: The text (`str`).
  """

  text: str


class _MlirTextProtocol(typing.Protocol):
  """Protocol for Lark lexer/parser inputs and nodes."""

  def __str__(self) -> str:
    """Return string representation.

    Returns:
        String representation.
    """
    ...


MlirChild: typing.TypeAlias = (
  Token | Tree[Token] | MlirNode | typing.Sequence[_MlirTextProtocol] | _MlirTextProtocol | str | None
)


@typing.runtime_checkable
class _ValueProtocol(typing.Protocol):
  """Protocol for tokens or nodes exposing a `value` attribute.

  Attributes:
      value: The value (`str | int | float | MlirNode | None`).
  """

  value: str | int | float | MlirNode | None


@typing.runtime_checkable
class _LeadingTriviaProtocol(typing.Protocol):
  """Protocol for tokens or nodes exposing `leading_trivia`.

  Attributes:
      leading_trivia: The leading trivia (`list[Trivia]`).
  """

  leading_trivia: list[Trivia]


def _tok_val(node: MlirChild) -> str:
  """Extract string value from a Token or child node.

  Args:
      node: Token or AST child.

  Returns:
      Extracted string value.
  """
  return str(node.value) if isinstance(node, _ValueProtocol) else ""


def _tree_data(node: MlirChild) -> str | None:
  """Extract rule name from a Lark Tree node.

  Args:
      node: Candidate Tree node.

  Returns:
      Rule name string if `node` is a `Tree`, else `None`.
  """
  return str(node.data) if isinstance(node, Tree) else None


def _tree_children(node: MlirChild) -> list[MlirChild]:
  """Extract children list from a Lark Tree node.

  Args:
      node: Candidate Tree node.

  Returns:
      List of child nodes if `node` is a `Tree`, else empty list.
  """
  return list(node.children) if isinstance(node, Tree) else []


class MlirToken(Token):
  """Custom token that carries its leading trivia.

  Attributes:
      leading_trivia: The leading trivia (`list[Trivia]`).
  """

  __slots__ = ("leading_trivia",)

  leading_trivia: list[Trivia]


class MlirLexer(Lexer):
  """Custom Lexer preserving trivia and matching MLIR tokens."""

  def __init__(self, lexer_conf: _MlirTextProtocol | None) -> None:
    """Init.

    Args:
        lexer_conf: Configuration for the lexer.
    """
    self.lexer_conf = lexer_conf

  def lex(self, lexer_state: _MlirTextProtocol, parser_state: _MlirTextProtocol | None = None) -> typing.Iterator[Token]:
    """Tokenize the input string and attach trivia.

    Args:
        lexer_state: The LexerState or string.
        parser_state: The ParserState.

    Yields:
        MlirToken: The next matched token carrying its leading trivia.

    Raises:
        ValueError: If an unexpected character token is encountered.
    """
    data = str(lexer_state.text) if isinstance(lexer_state, _LexerTextProtocol) else str(lexer_state)

    leading: list[Trivia] = []
    for mo in re.finditer(tok_regex, data):
      kind = mo.lastgroup or "MISMATCH"
      val = mo.group()
      if kind == "MISMATCH":
        raise SyntaxParseError(f"Unexpected '{val}'")
      if kind in ("WS", "COMMENT"):
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
            ".": "DOT",
          }
          kind = punct_map[val]

        t = MlirToken(kind, val)
        t.leading_trivia = list(leading)
        leading.clear()
        yield t


GRAMMAR = r"""
    ?start: module
    module: (operation | attribute_alias_def | type_alias_def)*

    operation: [results EQ] custom_operation [trailing_location] | [op_result_list] generic_operation [trailing_location]

    custom_operation: op_name [SYM_ID] [operands] [dictionary_attribute] op_tail*

    attribute_alias: ATTR_ALIAS_ID
    attribute_alias_def: attribute_alias EQ attribute_value

    ?op_tail: regions | COLON result_types | ARROW result_types

    results: VAL_ID (COMMA VAL_ID)*
    op_name: IDENTIFIER | STRING | SYM_ID
    bare_id_list: IDENTIFIER (COMMA IDENTIFIER)*

    operands: LPAREN [operand_list] RPAREN
            | LPAREN DOT DOT DOT RPAREN
            | operand_list
    operand_list: operand (COMMA operand)*

    operand: (value_use | SYM_ID) [COLON TYPE]

    dictionary_attribute: LBRACE [attribute_entry (COMMA attribute_entry)*] RBRACE
    attribute_entry: attr_name EQ attribute_value
    attr_name: IDENTIFIER | STRING
    attribute_value: STRING | NUMBER | decimal_literal | TYPE | type_list_parens | function_type | LBRACK [attribute_value (COMMA attribute_value)*] RBRACK | attribute_alias | IDENTIFIER | dialect_attribute | dense_attribute
    decimal_literal: NUMBER
    dense_attribute: IDENTIFIER OPAQUE_DIALECT_CONTENTS

    dialect_attribute: ATTR_ALIAS_ID (OPAQUE_DIALECT_CONTENTS | DOT IDENTIFIER)

    regions: region+
    region: LBRACE block* RBRACE

    block_label: block_id [block_arg_list] COLON
    block: [block_label] operation*
    block_arg_list: LPAREN [block_arg (COMMA block_arg)*] RPAREN
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
    generic_operation: STRING [LPAREN [value_use_list] RPAREN] [successor_list] [region_list] [dictionary_properties] [dictionary_attribute] [COLON function_type]
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
    successor: CARET_ID [COLON block_arg_list]
    successor_list: LBRACK successor (COMMA successor)* RBRACK
    trailing_location: IDENTIFIER LPAREN STRING RPAREN
    type_alias: TYPE
    type_alias_def: type_alias EQ TYPE
    type_list_no_parens: TYPE (COMMA TYPE)*
    type_list_parens: LPAREN [TYPE (COMMA TYPE)*] RPAREN
    value_id_and_type: VAL_ID COLON TYPE
    value_id_and_type_list: value_id_and_type (COMMA value_id_and_type)*
    value_id_list: VAL_ID (COMMA VAL_ID)*
    value_use: VAL_ID [ATTR_ALIAS_ID]
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


def _get_trivia(node: MlirChild | _LeadingTriviaProtocol | Tree[Token] | Token | str | int | None) -> list[Trivia]:
  """Extract leading trivia from a token or the first token in a tree.

  Args:
      node: The Lark AST Node or Token.

  Returns:
      A list of trivia items.
  """
  if isinstance(node, _LeadingTriviaProtocol):
    res = list(node.leading_trivia)
    node.leading_trivia = []
    return res
  if isinstance(node, Tree) and node.children:
    return _get_trivia(node.children[0])
  return []


class MlirTransformer(Transformer[Token, MlirNode]):
  """Transform parsed AST nodes into MlirNode classes."""

  @v_args(inline=False)
  def module(self, children: typing.Sequence[MlirChild]) -> ModuleNode:
    """Transform the top-level module rule.

    Args:
        children: Parsed children.

    Returns:
        The ModuleNode.
    """
    ops = [c for c in children if isinstance(c, OperationNode)]
    aliases: list[AliasNodeUnion] = [
      c
      for c in children
      if isinstance(c, (AttributeAliasDefNode, TypeAliasDefNode, MlirNode))
      and type(c).__name__ in ("AttributeAliasDefNode", "TypeAliasDefNode")
    ]
    leading = _get_trivia(children[0]) if children else []
    return ModuleNode(body=BlockNode(label="", operations=ops), aliases=aliases, leading_trivia=leading)

  @v_args(inline=False)
  def type_alias_def(self, children: typing.Sequence[MlirChild]) -> "TypeAliasDefNode":
    """Transform type_alias_def.

    Args:
        children: Parsed children.

    Returns:
        The TypeAliasDefNode.
    """
    c0_children = _tree_children(children[0])
    name_token = c0_children[0]
    name = _tok_val(name_token).lstrip("!")
    type_node = TypeNode(body=_tok_val(children[2]), leading_trivia=_get_trivia(children[2]))
    leading = _get_trivia(children[0])
    trailing = []
    if len(children) > 1 and _tree_data(children[-1]) == "trivia":
      trailing = _get_trivia(children[-1])
    return TypeAliasDefNode(name=name, type_node=type_node, leading_trivia=leading, trailing_trivia=trailing)

  @v_args(inline=False)
  def attribute_alias_def(self, children: typing.Sequence[MlirChild]) -> "AttributeAliasDefNode":
    """Transform an attribute alias definition.

    Args:
        children: Parsed children.

    Returns:
        The AttributeAliasDefNode.
    """
    # ATTR_ALIAS_ID trivia? "=" trivia? attribute_value trivia?
    c0_children = _tree_children(children[0])
    name_token = c0_children[0]
    name = _tok_val(name_token).lstrip("!")

    # Find attribute_value
    val_node = next(c for c in children if _tree_data(c) == "attribute_value")
    val_children = _tree_children(val_node)
    if len(val_children) == 1:
      val = val_children[0]
      val_str = str(val.value) if isinstance(val, Token) else str(val)
    else:
      # Array of values or something else
      val_str = str(val_node)

    leading = _get_trivia(children[0])

    # Handle trivia on trailing elements
    trailing = []
    if len(children) > 1 and _tree_data(children[-1]) == "trivia":
      trailing = _get_trivia(children[-1])

    # The "=" token is not captured, trivia after "=" is part of children if it matches trivia rule.
    # To keep it simple, we store raw string for now.

    return AttributeAliasDefNode(name=name, value_str=val_str, leading_trivia=leading, trailing_trivia=trailing)

  def _parse_op_results(self, op: OperationNode, child: MlirChild, rule: str) -> None:
    """Populate ``op.results`` from ``op_result_list`` or ``results`` trees.

    Args:
        op: Target operation node being constructed.
        child: Parsed result list subtree.
        rule: Grammar rule name (``'op_result_list'`` or ``'results'``).
    """
    if rule == "op_result_list":
      for item in _tree_children(child):
        if _tree_data(item) == "op_result":
          v_tok = _tree_children(item)[0]
          op.results.append(ValueNode(name=_tok_val(v_tok), leading_trivia=_get_trivia(v_tok)))
    else:
      op.results = [
        ValueNode(name=_tok_val(v), leading_trivia=_get_trivia(v))
        for v in _tree_children(child)
        if isinstance(v, Token) and v.type == "VAL_ID"
      ]

  def _parse_single_operand(self, op_val: MlirChild) -> ValueNode:
    """Convert a single ``operand`` subtree into a ``ValueNode``.

    Args:
        op_val: Parsed ``operand`` subtree.

    Returns:
        Constructed ``ValueNode`` with optional use index and type annotation.
    """
    op_val_children = _tree_children(op_val)
    v_node = op_val_children[0]
    idx: int | None = None
    if isinstance(v_node, Token):
      v_tok: MlirChild = v_node
    else:
      v_node_children = _tree_children(v_node)
      v_tok = v_node_children[0]
      if len(v_node_children) > 1 and v_node_children[1] is not None:
        idx = int(_tok_val(v_node_children[1]).lstrip("#"))

    type_node: TypeNode | None = None
    colon_triv: list[Trivia] = []
    if len(op_val_children) > 2 and isinstance(op_val_children[2], Token) and op_val_children[2].type == "TYPE":
      colon_triv = _get_trivia(op_val_children[1])
      type_node = TypeNode(
        body=_tok_val(op_val_children[2]),
        leading_trivia=_get_trivia(op_val_children[2]),
      )
    return ValueNode(
      name=_tok_val(v_tok),
      use_index=idx,
      leading_trivia=_get_trivia(v_tok),
      type_node=type_node,
      colon_trivia=colon_triv,
    )

  def _parse_operands_tree(self, op: OperationNode, child: MlirChild) -> None:
    """Populate ``op.has_parens`` and ``op.operands`` from an ``operands`` subtree.

    Args:
        op: Target operation node being constructed.
        child: Parsed ``operands`` subtree.
    """
    c_children = _tree_children(child)
    op.has_parens = bool(c_children and isinstance(c_children[0], Token) and c_children[0].type == "LPAREN")
    operand_list_node = next((val for val in c_children if _tree_data(val) == "operand_list"), None)
    if operand_list_node is not None:
      for op_val in _tree_children(operand_list_node):
        if _tree_data(op_val) == "operand":
          op.operands.append(self._parse_single_operand(op_val))

  def _parse_op_tail(self, op: OperationNode, child: MlirChild) -> None:
    """Populate tail separator trivia and result types from an ``op_tail`` subtree.

    Args:
        op: Target operation node being constructed.
        child: Parsed ``op_tail`` subtree.
    """
    c_children = _tree_children(child)
    if not c_children:
      return
    op.op_tail_str = _tok_val(c_children[0])
    op.op_tail_trivia = _get_trivia(c_children[0])
    for tail_child in c_children:
      if _tree_data(tail_child) == "result_types":
        for t in _tree_children(tail_child):
          if isinstance(t, Token) and t.type == "TYPE":
            op.result_types.append(TypeNode(body=_tok_val(t), leading_trivia=_get_trivia(t)))

  def _parse_custom_op(self, op: OperationNode, child: MlirChild, rule: str | None) -> bool:
    """Handle custom-operation name, symbol reference, operand list, or tail productions.

    Args:
        op: Target operation node being constructed.
        child: Current parsed child node or token.
        rule: Grammar rule name of ``child``, if it is a ``Tree``.

    Returns:
        True if ``child`` matched a custom-operation production, otherwise False.
    """
    if rule == "op_name":
      op.name = _tok_val(_tree_children(child)[0])
      return True
    if isinstance(child, Token) and child.type == "SYM_ID":
      existing_triv = list(op.name_trivia) if isinstance(op.name_trivia, list) else []
      existing_triv.extend(_get_trivia(child))
      existing_triv.append(Trivia(child.value))
      op.name_trivia = existing_triv
      return True
    if rule == "operands":
      self._parse_operands_tree(op, child)
      return True
    if rule == "op_tail":
      self._parse_op_tail(op, child)
      return True
    return False

  def _parse_generic_op(self, op: OperationNode, child: MlirChild, rule: str | None) -> bool:
    """Handle generic-operation string name, ``value_use_list``, and ``successor_list`` productions.

    Args:
        op: Target operation node being constructed.
        child: Current parsed child node or token.
        rule: Grammar rule name of ``child``, if it is a ``Tree``.

    Returns:
        True if ``child`` matched a generic-operation production, otherwise False.
    """
    if isinstance(child, Token) and child.type == "STRING" and not op.name:
      op.name = child.value.strip('"')
      op.is_generic = True
      op.name_trivia = _get_trivia(child)
      return True
    if rule == "value_use_list":
      for op_val in _tree_children(child):
        if _tree_data(op_val) == "value_use":
          v_tok = _tree_children(op_val)[0]
          op.operands.append(ValueNode(name=_tok_val(v_tok), leading_trivia=_get_trivia(v_tok)))
      return True
    if rule == "successor_list":
      for succ in _tree_children(child):
        if _tree_data(succ) == "successor":
          caret = _tree_children(succ)[0]
          if _tree_data(caret) == "caret_id":
            caret_tok = _tree_children(caret)[1]
            op.successors.append("^" + _tok_val(caret_tok))
          elif isinstance(caret, Token) and caret.type == "CARET_ID":
            op.successors.append(caret.value)
      return True
    return False

  def _parse_regions(self, op: OperationNode, child: MlirChild, rule: str | None) -> bool:
    """Populate ``op.regions`` from a ``region_list`` tree or pre-transformed ``RegionNode`` list.

    Args:
        op: Target operation node being constructed.
        child: Current parsed child node or region list.
        rule: Grammar rule name of ``child``, if it is a ``Tree``.

    Returns:
        True if ``child`` represented regions, otherwise False.
    """
    if rule == "region_list":
      for reg in _tree_children(child):
        if _tree_data(reg) == "region":
          op.regions.append(self.region([reg]))
      return True
    if isinstance(child, list) and len(child) > 0 and isinstance(child[0], RegionNode):
      op.regions.extend([r for r in child if isinstance(r, RegionNode)])
      return True
    return False

  def _parse_attribute_dict(self, op: OperationNode, child: MlirChild, rule: str | None) -> bool:
    """Populate ``op.attributes`` or ``op.properties`` from attribute/property structures.

    Args:
        op: Target operation node being constructed.
        child: Current parsed child node or attribute list.
        rule: Grammar rule name of ``child``, if it is a ``Tree``.

    Returns:
        True if ``child`` represented attributes or properties, otherwise False.
    """
    if isinstance(child, list) and len(child) > 0 and isinstance(child[0], AttributeNode):
      op.attributes = [a for a in child if isinstance(a, AttributeNode)]
      return True
    if rule == "dictionary_properties":
      for prop_child in _tree_children(child):
        if isinstance(prop_child, list):
          op.properties = [p for p in prop_child if isinstance(p, AttributeNode)]
      return True
    return False

  @staticmethod
  def _collect_signature_types(parts: typing.Sequence[MlirChild]) -> list[TypeNode]:
    """Extract ordered ``TypeNode`` entries from a ``function_type`` segment.

    Args:
        parts: Sequence of ``TYPE`` tokens or ``type_list_parens`` subtrees.

    Returns:
        Ordered list of parsed ``TypeNode`` instances.
    """
    collected: list[TypeNode] = []
    for pt in parts:
      if isinstance(pt, Token) and pt.type == "TYPE":
        collected.append(TypeNode(body=_tok_val(pt), leading_trivia=_get_trivia(pt)))
      elif _tree_data(pt) == "type_list_parens":
        collected.extend(
          TypeNode(body=_tok_val(inner_pt), leading_trivia=_get_trivia(inner_pt))
          for inner_pt in _tree_children(pt)
          if isinstance(inner_pt, Token) and inner_pt.type == "TYPE"
        )
    return collected

  def _parse_function_signature(self, op: OperationNode, child: MlirChild) -> None:
    """Apply operand and result types from a ``function_type`` tree onto ``op``.

    Args:
        op: Target operation node being constructed.
        child: Parsed ``function_type`` subtree.
    """
    parts = _tree_children(child)
    arrow_idx = next(
      (idx for idx, pt in enumerate(parts) if isinstance(pt, Token) and pt.type == "ARROW"),
      -1,
    )
    if arrow_idx == -1:
      return
    op.result_types.extend(self._collect_signature_types(parts[arrow_idx + 1 :]))
    for arg_idx, arg_type in enumerate(self._collect_signature_types(parts[:arrow_idx])):
      if arg_idx < len(op.operands):
        op.operands[arg_idx].type_node = arg_type

  def _apply_op_child(self, op: OperationNode, child: MlirChild, rule: str | None) -> None:
    """Dispatch a single flattened operation child to its production handler.

    Args:
        op: Target operation node being constructed.
        child: Current parsed child node or token.
        rule: Grammar rule name of ``child``, if it is a ``Tree``.
    """
    if rule == "op_result_list" or rule == "results":
      self._parse_op_results(op, child, rule)
    elif (
      self._parse_custom_op(op, child, rule)
      or self._parse_generic_op(op, child, rule)
      or self._parse_regions(op, child, rule)
      or self._parse_attribute_dict(op, child, rule)
    ):
      return
    elif rule == "function_type":
      self._parse_function_signature(op, child)
    elif rule == "trailing_location":
      op.location = _tok_val(_tree_children(child)[2])

  @v_args(inline=False)
  def operation(self, children: list[MlirChild]) -> OperationNode:
    """Transform an operation rule into an OperationNode.

    Args:
        children: Parsed children.

    Returns:
        The constructed OperationNode.
    """
    op = OperationNode()
    first_non_none = next((c for c in children if c is not None), None)
    if first_non_none is not None:
      op.leading_trivia = _get_trivia(first_non_none)

    items = list(children)
    i = 0
    while i < len(items):
      c = items[i]
      rule = _tree_data(c)
      if rule in ("custom_operation", "generic_operation"):
        items = items[:i] + _tree_children(c) + items[i + 1 :]
        continue
      self._apply_op_child(op, c, rule)
      i += 1
    return op

  @v_args(inline=False)
  def dictionary_attribute(self, children: typing.Sequence[MlirChild]) -> list[AttributeNode]:
    """Transform the dictionary_attribute rule into a list of AttributeNode.

    Args:
        children: Parsed children.

    Returns:
        List of parsed attributes.
    """
    attrs = []
    for c in children:
      if _tree_data(c) == "attribute_entry":
        c_children = _tree_children(c)
        name = _tok_val(_tree_children(c_children[0])[0])
        val_node = c_children[2]
        val_children = _tree_children(val_node)

        val: str | list[str]
        if len(val_children) == 1:
          first_val = val_children[0]
          if isinstance(first_val, Tree):
            val = "".join(_tok_val(t) for t in first_val.scan_values(lambda v: isinstance(v, Token)))
          else:
            val = _tok_val(first_val)
        else:
          # Array of values
          val = [_tok_val(_tree_children(v)[0]) for v in val_children if _tree_data(v) == "attribute_value"]

        attr = AttributeNode(name=name, value=val, leading_trivia=_get_trivia(c_children[0]))
        attrs.append(attr)
    return attrs

  @v_args(inline=False)
  def region(self, children: typing.Sequence[MlirChild]) -> RegionNode:
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
  def regions(self, children: typing.Sequence[MlirChild]) -> list[RegionNode]:
    """Transform the regions rule into a list of RegionNode.

    Args:
        children: Parsed children.

    Returns:
        List of parsed regions.
    """
    return [c for c in children if isinstance(c, RegionNode)]

  @v_args(inline=False)
  def block(self, children: typing.Sequence[MlirChild]) -> BlockNode:
    """Transform the block rule into a BlockNode.

    Args:
        children: Parsed children.

    Returns:
        The constructed BlockNode.
    """
    label = ""
    args: list[tuple[ValueNode, TypeNode]] = []
    ops: list[OperationNode] = []
    for c in children:
      if _tree_data(c) == "block_label":
        for child in _tree_children(c):
          if _tree_data(child) == "block_id":
            label = _tok_val(_tree_children(child)[0])
          elif _tree_data(child) == "block_arg_list":
            for arg in _tree_children(child):
              if _tree_data(arg) == "block_arg":
                arg_children = _tree_children(arg)
                v = ValueNode(
                  name=_tok_val(arg_children[0]),
                  leading_trivia=_get_trivia(arg_children[0]),
                )
                t = TypeNode(
                  body=_tok_val(arg_children[2]),
                  leading_trivia=_get_trivia(arg_children[2]),
                )
                args.append((v, t))
      elif isinstance(c, OperationNode):
        ops.append(c)

    leading = _get_trivia(children[0]) if children else []
    return BlockNode(label=label, arguments=args, operations=ops, leading_trivia=leading)


_CACHED_PARSER = None


class MlirParser:
  """Parse a stream of MLIR tokens into a Concrete Syntax Tree."""

  def __init__(self, text: str):
    """Initialize the parser.

    Args:
        text (str): The MLIR source code to parse.
    """
    self.text = text
    global _CACHED_PARSER
    if _CACHED_PARSER is None:
      _CACHED_PARSER = Lark(GRAMMAR, parser="earley", lexer=MlirLexer)
    self.parser = _CACHED_PARSER
    self.transformer = MlirTransformer()

  def parse(self) -> ModuleNode:
    """Top-level parsing entry point.

    Returns:
        ModuleNode: The root of the MLIR CST.

    Raises:
        SyntaxParseError: If the MLIR input fails Lark parsing or root validation.
    """
    if not self.text.strip():
      return ModuleNode()

    try:
      tree = self.parser.parse(self.text)
      node = self.transformer.transform(tree)
    except LarkError as exc:
      raise SyntaxParseError(f"Failed to parse MLIR: {exc}") from exc
    if not isinstance(node, ModuleNode):
      raise SyntaxParseError(f"Expected ModuleNode root, got {type(node).__name__}")
    return node


__all__ = [
  "AttributeAliasDefNode",
  "AttributeNode",
  "BlockNode",
  "MlirLexer",
  "MlirParser",
  "MlirToken",
  "MlirTransformer",
  "ModuleNode",
  "OperationNode",
  "RegionNode",
  "Trivia",
  "TypeAliasDefNode",
  "TypeNode",
  "ValueNode",
]
