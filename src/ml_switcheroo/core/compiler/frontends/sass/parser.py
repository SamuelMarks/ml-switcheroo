"""SASS Parser Implementation.

This module provides the `SassParser`, a pure Python Lark-based parser
that converts a stream of characters into a Concrete Syntax Tree defined in `cst.py`.
"""

import re
from typing import List, Any, cast
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassDirective,
  SassImmediate,
  SassInstruction,
  SassLabel,
  SassMemory,
  SassOperand,
  SassPredicate,
  SassRegister,
  SassModule,
)
from ml_switcheroo.core.cst.base import Trivia

from lark import Lark, Transformer, v_args
from lark.lexer import Lexer, Token

TOKEN_REGEX = [
  ("COMMENT", r"//[^\n]*"),
  ("WS", r"[ \t\f\r\n]+"),
  ("HEX_NUMBER", r"-?0x[0-9a-fA-F]+"),
  ("NUMBER", r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"),
  ("STRING", r'"(?:[^"\\]|\\.)*"'),
  ("REG_IDENTIFIER", r"[-|]*[RU]?[RZS]\d+\|?|PT|RZ|RZS"),
  ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
  ("PUNCTUATION", r"[;:,\.\[\]\+\-!@]"),
  ("MISMATCH", r"."),
]
tok_regex = "|".join("(?P<%s>%s)" % pair for pair in TOKEN_REGEX)


class SassToken(Token):
  """Custom token that carries its leading trivia."""

  __slots__ = ("leading_trivia",)

  leading_trivia: List[Trivia]


class SassLexer(Lexer):
  """Custom Lexer preserving trivia and matching SASS tokens."""

  def __init__(self, lexer_conf: Any):
    """Initialize the SassLexer.

    Args:
        lexer_conf: The Lark lexer configuration.
    """
    self.lexer_conf = lexer_conf

  def lex(self, data: str) -> Any:  # type: ignore[override]
    """Tokenize the input string and attach trivia.

    Args:
        data: The raw input SASS string to tokenize.

    Returns:
        An iterator of SassToken instances.
    """
    leading: List[Trivia] = []
    for mo in re.finditer(tok_regex, data):
      kind = mo.lastgroup
      val = mo.group()
      if kind == "MISMATCH":
        raise ValueError(f"Unexpected '{val}'")
      if kind == "WS":
        assert val is not None
        leading.append(Trivia(val))
      elif kind == "COMMENT":
        assert val is not None
        t = SassToken("COMMENT", val)
        t.leading_trivia = list(leading)
        leading.clear()
        yield t
      else:
        if kind == "PUNCTUATION":
          punct_map = {
            ";": "SEMI",
            ":": "COLON",
            ",": "COMMA",
            ".": "DOT",
            "[": "LBRACK",
            "]": "RBRACK",
            "+": "PLUS",
            "-": "MINUS",
            "!": "BANG",
            "@": "AT",
          }
          assert val is not None
          kind = punct_map[val]

        assert kind is not None
        assert val is not None
        t2 = SassToken(kind, val)
        t2.leading_trivia = list(leading)
        leading.clear()
        yield t2


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
    return _get_trivia(node.children[0])
  return []


GRAMMAR = r"""
    ?start: module
    module: statement*

    ?statement: instruction | directive | label | empty_statement | comment_stmt

    empty_statement: SEMI
    comment_stmt: COMMENT

    directive: DOT IDENTIFIER [directive_params]
    directive_params: param_list
    param_list: directive_param (COMMA directive_param)*
    ?directive_param: STRING | IDENTIFIER | AT STRING -> at_string | NUMBER | HEX_NUMBER

    label: IDENTIFIER COLON

    instruction: [predicate] IDENTIFIER [operands] [SEMI]

    predicate: AT [BANG] (IDENTIFIER | REG_IDENTIFIER)

    operands: operand (COMMA operand)*

    ?operand: memory | register | immediate | predicate_operand | identifier

    memory: IDENTIFIER LBRACK HEX_NUMBER RBRACK LBRACK HEX_NUMBER RBRACK -> mem_bank
          | IDENTIFIER LBRACK HEX_NUMBER RBRACK -> mem_bank_single
          | LBRACK register RBRACK -> mem_reg
          | LBRACK register PLUS immediate_val RBRACK -> mem_reg_offset
          | LBRACK register MINUS immediate_val RBRACK -> mem_reg_neg_offset

    register: REG_IDENTIFIER

    predicate_operand: AT BANG IDENTIFIER -> pred_at_bang_id
                     | AT BANG REG_IDENTIFIER -> pred_at_bang_reg
                     | AT IDENTIFIER -> pred_at_id
                     | AT REG_IDENTIFIER -> pred_at_reg
                     | BANG IDENTIFIER -> pred_bang_id
                     | BANG REG_IDENTIFIER -> pred_bang_reg

    immediate: NUMBER | HEX_NUMBER
    ?immediate_val: NUMBER | HEX_NUMBER
    identifier: IDENTIFIER

    SEMI: ";"
    COLON: ":"
    COMMA: ","
    DOT: "."
    LBRACK: "["
    RBRACK: "]"
    PLUS: "+"
    MINUS: "-"
    BANG: "!"
    AT: "@"

    IDENTIFIER: /.+/
    REG_IDENTIFIER: /.+/
    NUMBER: /.+/
    HEX_NUMBER: /.+/
    STRING: /.+/
    COMMENT: /.+/
"""


class SassTransformer(Transformer[Any, Any]):
  """Transforms parsed AST nodes into SassNode classes."""

  @v_args(inline=False)
  def module(self, children: List[Any]) -> SassModule:
    """Transform the top-level module rule.

    Args:
        children: Parsed children.

    Returns:
        The SassModule.
    """
    leading = _get_trivia(children[0]) if children else []
    mod = SassModule(statements=[c for c in children if c is not None])
    mod.leading_trivia = leading
    return mod

  @v_args(inline=False)
  def empty_statement(self, children: List[Any]) -> None:
    """Transform an empty statement into None.

    Args:
        children: Parsed children.

    Returns:
        None.
    """
    return None

  @v_args(inline=False)
  def comment_stmt(self, children: List[Any]) -> SassComment:
    """Transform a comment into a SassComment.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassComment.
    """
    c = SassComment(text=children[0].value[2:].strip())
    c.leading_trivia = _get_trivia(children[0])
    return c

  @v_args(inline=False)
  def directive(self, children: List[Any]) -> SassDirective:
    """Transform a directive into a SassDirective.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassDirective.
    """
    name = children[1].value
    params: List[str] = []
    if len(children) > 2 and children[2] is not None:
      # children[2] is directive_params Tree, its child is param_list result (which is a list)
      param_list = children[2].children[0] if getattr(children[2], "data", None) == "directive_params" else children[2]
      if isinstance(param_list, list):
        for p in param_list:
          if getattr(p, "type", None) == "COMMA":
            continue
          if getattr(p, "data", None) == "at_string":
            params.append("".join(getattr(x, "value", "") for x in p.children))
          elif isinstance(p, list):
            params.append("".join(getattr(x, "value", "") for x in p))
          elif isinstance(p, Token):
            params.append(p.value)
          else:
            params.append(str(p))
      else:
        params.append(str(param_list))
    d = SassDirective(name=name, params=params)
    d.leading_trivia = _get_trivia(children[0])
    return d

  @v_args(inline=False)
  def at_string(self, children: List[Any]) -> List[Any]:
    """Transform an AT STRING into a list of Tokens.

    Args:
        children: Parsed children.

    Returns:
        The children list.
    """
    return children

  @v_args(inline=False)
  def param_list(self, children: List[Any]) -> List[Any]:
    """Transform a parameter list.

    Args:
        children: Parsed children.

    Returns:
        The children list.
    """
    return children

  @v_args(inline=False)
  def label(self, children: List[Any]) -> SassLabel:
    """Transform a label into a SassLabel.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassLabel.
    """
    lbl = SassLabel(name=children[0].value)
    lbl.leading_trivia = _get_trivia(children[0])
    return lbl

  @v_args(inline=False)
  def instruction(self, children: List[Any]) -> SassInstruction:
    """Transform an instruction.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassInstruction.
    """
    predicate = None
    opcode = ""
    operands = []

    leading = None
    for c in children:
      if c is not None and leading is None:
        leading = _get_trivia(c)
        break

    for c in children:
      if isinstance(c, SassPredicate):
        predicate = c
      elif isinstance(c, Token) and c.type == "IDENTIFIER":
        opcode = c.value
      elif getattr(c, "type", None) == "SEMI":
        pass
      elif isinstance(c, list):
        operands = c

    i = SassInstruction(opcode=opcode, operands=operands, predicate=predicate)
    i.leading_trivia = leading if leading else []
    return i

  @v_args(inline=False)
  def predicate(self, children: List[Any]) -> SassPredicate:
    """Transform a predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassPredicate.
    """
    negated = False
    name = ""
    leading = _get_trivia(children[0])
    for c in children:
      if getattr(c, "type", None) == "BANG":
        negated = True
      elif getattr(c, "type", None) in ["IDENTIFIER", "REG_IDENTIFIER"]:
        name = c.value
    p = SassPredicate(name=name, negated=negated, is_guard=True)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def operands(self, children: List[Any]) -> List[SassOperand]:
    """Transform an operands list.

    Args:
        children: Parsed children.

    Returns:
        The children list.
    """
    return [c for c in children if isinstance(c, SassOperand)]

  @v_args(inline=False)
  def mem_bank(self, children: List[Any]) -> SassMemory:
    """Transform a memory bank access.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassMemory.
    """
    leading = _get_trivia(children[0])
    bank = int(children[2].value, 16)
    offset = int(children[5].value, 16)
    m = SassMemory(base=f"{children[0].value}[{hex(bank)}]", offset=offset)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_bank_single(self, children: List[Any]) -> SassMemory:
    """Transform a memory bank access without offset.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassMemory.
    """
    leading = _get_trivia(children[0])
    bank = int(children[2].value, 16)
    m = SassMemory(base=f"{children[0].value}[{hex(bank)}]", offset=None)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg(self, children: List[Any]) -> SassMemory:
    """Transform a memory register access.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassMemory.
    """
    leading = _get_trivia(children[0])
    m = SassMemory(base=children[1], offset=None)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg_offset(self, children: List[Any]) -> SassMemory:
    """Transform a memory register access with offset.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassMemory.
    """
    leading = _get_trivia(children[0])
    offset_tok = children[3].children[0] if getattr(children[3], "children", None) else children[3]
    offset = int(offset_tok.value, 16 if "0x" in offset_tok.value.lower() else 10)
    m = SassMemory(base=children[1], offset=offset)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg_neg_offset(self, children: List[Any]) -> SassMemory:
    """Transform a memory register access with negative offset.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassMemory.
    """
    leading = _get_trivia(children[0])
    offset_tok = children[3].children[0] if getattr(children[3], "children", None) else children[3]
    offset = -int(offset_tok.value, 16 if "0x" in offset_tok.value.lower() else 10)
    m = SassMemory(base=children[1], offset=offset)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def register(self, children: List[Any]) -> SassRegister:
    """Transform a register.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassRegister.
    """
    leading = _get_trivia(children[0])
    val = children[0].value
    negated = val.startswith("-") or val.startswith("|-")
    absolute = "|" in val

    name = val.replace("-", "").replace("|", "")
    r = SassRegister(name=name, negated=negated, absolute=absolute)
    r.leading_trivia = leading
    return r

  @v_args(inline=False)
  def pred_at_bang_id(self, children: List[Any]) -> SassPredicate:
    """Transform a @!identifier predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassPredicate.
    """
    leading = _get_trivia(children[0])
    p = SassPredicate(name=children[2].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_at_bang_reg(self, children: List[Any]) -> SassPredicate:
    """Transform a @!register predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassPredicate.
    """
    leading = _get_trivia(children[0])
    p = SassPredicate(name=children[2].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_at_id(self, children: List[Any]) -> SassPredicate:
    """Transform a @identifier predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassPredicate.
    """
    leading = _get_trivia(children[0])
    p = SassPredicate(name=children[1].value, negated=False, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_at_reg(self, children: List[Any]) -> SassPredicate:
    """Transform a @register predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassPredicate.
    """
    leading = _get_trivia(children[0])
    p = SassPredicate(name=children[1].value, negated=False, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_bang_id(self, children: List[Any]) -> SassPredicate:
    """Transform a !identifier predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassPredicate.
    """
    leading = _get_trivia(children[0])
    p = SassPredicate(name=children[1].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_bang_reg(self, children: List[Any]) -> SassPredicate:
    """Transform a !register predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassPredicate.
    """
    leading = _get_trivia(children[0])
    p = SassPredicate(name=children[1].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def immediate(self, children: List[Any]) -> SassImmediate:
    """Transform an immediate value.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassImmediate.
    """
    leading = _get_trivia(children[0])
    val_str = children[0].value
    is_hex = "0x" in val_str.lower()
    if is_hex:
      val = cast(Any, int(val_str, 16))
    else:
      val2 = float(val_str) if "." in val_str else int(val_str, 10)
      val = cast(Any, val2)
    i = SassImmediate(value=val, is_hex=is_hex)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def identifier(self, children: List[Any]) -> SassLabel:
    """Transform an identifier operand.

    Args:
        children: Parsed children.

    Returns:
        The parsed SassLabel.
    """
    leading = _get_trivia(children[0])
    lbl = SassLabel(name=children[0].value)
    lbl.leading_trivia = leading
    return lbl


class SassParser:
  """Facade for parsing SASS strings into CST modules."""

  def __init__(self, code: str) -> None:
    """Initialize the parser with the SASS source code.

    Args:
        code: The raw SASS string.
    """
    self.code = code
    self.parser = Lark(GRAMMAR, parser="earley", lexer=SassLexer)
    self.transformer = SassTransformer()

  def parse(self) -> SassModule:
    """Parse the entire code block.

    Returns:
        SassModule: The root CST node.
    """
    if not self.code.strip():
      return SassModule()

    try:
      tree = self.parser.parse(self.code)
      return self.transformer.transform(tree)  # type: ignore
    except Exception as e:
      raise ValueError(f"Unexpected token: {e}")
