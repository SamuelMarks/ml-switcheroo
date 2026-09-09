"""NVIDIA_SASS Parser Implementation.

This module provides the `NvidiaSassParser`, a pure Python Lark-based parser
that converts a stream of characters into a Concrete Syntax Tree defined in `cst.py`.
"""

import re
from typing import List, cast, Any
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassOperand,
  NvidiaSassPredicate,
  NvidiaSassRegister,
  NvidiaSassModule,
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


class NvidiaSassToken(Token):
  """Custom token that carries its leading trivia."""

  __slots__ = ("leading_trivia",)

  leading_trivia: List[Trivia]


class NvidiaSassLexer(Lexer):
  """Custom Lexer preserving trivia and matching NVIDIA_SASS tokens."""

  def __init__(self, lexer_conf: Any) -> None:
    """Initialize the NvidiaSassLexer.

    Args:
        lexer_conf: The Lark lexer configuration.
    """
    self.lexer_conf = lexer_conf

  def lex(self, lexer_state: Any, parser_state: Any = None) -> Any:
    """Tokenize the input string and attach trivia.

    Args:
        lexer_state: The raw input NVIDIA_SASS string or lexer state to tokenize.
        parser_state: Optional parser state.

    Returns:
        An iterator of NvidiaSassToken instances.
    """
    if isinstance(lexer_state, str):
      data = lexer_state
    elif hasattr(lexer_state, "text"):
      data = str(lexer_state.text)
    else:
      data = str(lexer_state)
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
        t = NvidiaSassToken("COMMENT", val)
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
        t2 = NvidiaSassToken(kind, val)
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


class NvidiaSassTransformer(Transformer[Any, Any]):
  """Transform parsed AST nodes into NvidiaSassNode classes."""

  @v_args(inline=False)
  def module(self, children: list[Any]) -> NvidiaSassModule:
    """Transform the top-level module rule.

    Args:
        children: Parsed children.

    Returns:
        The NvidiaSassModule.
    """
    leading = _get_trivia(children[0]) if children else []
    mod = NvidiaSassModule(statements=[c for c in children if c is not None])
    mod.leading_trivia = leading
    return mod

  @v_args(inline=False)
  def empty_statement(self, children: list[Any]) -> None:
    """Transform an empty statement into None.

    Args:
        children: Parsed children.

    Returns:
        None.
    """
    return None

  @v_args(inline=False)
  def comment_stmt(self, children: list[Any]) -> NvidiaSassComment:
    """Transform a comment into a NvidiaSassComment.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassComment.
    """
    c = NvidiaSassComment(text=children[0].value[2:].strip())
    c.leading_trivia = _get_trivia(children[0])
    return c

  @v_args(inline=False)
  def directive(self, children: list[Any]) -> NvidiaSassDirective:
    """Transform a directive into a NvidiaSassDirective.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassDirective.
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
    d = NvidiaSassDirective(name=name, params=params)
    d.leading_trivia = _get_trivia(children[0])
    return d

  @v_args(inline=False)
  def at_string(self, children: list[Any]) -> Any:
    """Transform an AT STRING into a list of Tokens.

    Args:
        children: Parsed children.

    Returns:
        The children list.
    """
    return children

  @v_args(inline=False)
  def param_list(self, children: list[Any]) -> Any:
    """Transform a parameter list.

    Args:
        children: Parsed children.

    Returns:
        The children list.
    """
    return children

  @v_args(inline=False)
  def label(self, children: list[Any]) -> NvidiaSassLabel:
    """Transform a label into a NvidiaSassLabel.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassLabel.
    """
    lbl = NvidiaSassLabel(name=children[0].value)
    lbl.leading_trivia = _get_trivia(children[0])
    return lbl

  @v_args(inline=False)
  def instruction(self, children: list[Any]) -> NvidiaSassInstruction:
    """Transform an instruction.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassInstruction.
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
      if isinstance(c, NvidiaSassPredicate):
        predicate = c
      elif isinstance(c, Token) and c.type == "IDENTIFIER":
        opcode = c.value
      elif getattr(c, "type", None) == "SEMI":
        pass
      elif isinstance(c, list):
        operands = c

    i = NvidiaSassInstruction(opcode=opcode, operands=operands, predicate=predicate)
    i.leading_trivia = leading if leading else []
    return i

  @v_args(inline=False)
  def predicate(self, children: list[Any]) -> NvidiaSassPredicate:
    """Transform a predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassPredicate.
    """
    negated = False
    name = ""
    leading = _get_trivia(children[0])
    for c in children:
      if getattr(c, "type", None) == "BANG":
        negated = True
      elif getattr(c, "type", None) in ["IDENTIFIER", "REG_IDENTIFIER"]:
        name = c.value
    p = NvidiaSassPredicate(name=name, negated=negated, is_guard=True)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def operands(self, children: list[Any]) -> List[NvidiaSassOperand]:
    """Transform an operands list.

    Args:
        children: Parsed children.

    Returns:
        The children list.
    """
    return [c for c in children if isinstance(c, NvidiaSassOperand)]

  @v_args(inline=False)
  def mem_bank(self, children: list[Any]) -> NvidiaSassMemory:
    """Transform a memory bank access.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassMemory.
    """
    leading = _get_trivia(children[0])
    bank = int(children[2].value, 16)
    offset = int(children[5].value, 16)
    m = NvidiaSassMemory(base=f"{children[0].value}[{hex(bank)}]", offset=offset)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_bank_single(self, children: list[Any]) -> NvidiaSassMemory:
    """Transform a memory bank access without offset.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassMemory.
    """
    leading = _get_trivia(children[0])
    bank = int(children[2].value, 16)
    m = NvidiaSassMemory(base=f"{children[0].value}[{hex(bank)}]", offset=None)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg(self, children: list[Any]) -> NvidiaSassMemory:
    """Transform a memory register access.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassMemory.
    """
    leading = _get_trivia(children[0])
    m = NvidiaSassMemory(base=children[1], offset=None)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg_offset(self, children: list[Any]) -> NvidiaSassMemory:
    """Transform a memory register access with offset.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassMemory.
    """
    leading = _get_trivia(children[0])
    offset_tok = children[3].children[0] if getattr(children[3], "children", None) else children[3]
    offset = int(offset_tok.value, 16 if "0x" in offset_tok.value.lower() else 10)
    m = NvidiaSassMemory(base=children[1], offset=offset)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg_neg_offset(self, children: list[Any]) -> NvidiaSassMemory:
    """Transform a memory register access with negative offset.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassMemory.
    """
    leading = _get_trivia(children[0])
    offset_tok = children[3].children[0] if getattr(children[3], "children", None) else children[3]
    offset = -int(offset_tok.value, 16 if "0x" in offset_tok.value.lower() else 10)
    m = NvidiaSassMemory(base=children[1], offset=offset)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def register(self, children: list[Any]) -> NvidiaSassRegister:
    """Transform a register.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassRegister.
    """
    leading = _get_trivia(children[0])
    val = children[0].value
    negated = val.startswith("-") or val.startswith("|-")
    absolute = "|" in val

    name = val.replace("-", "").replace("|", "")
    r = NvidiaSassRegister(name=name, negated=negated, absolute=absolute)
    r.leading_trivia = leading
    return r

  @v_args(inline=False)
  def pred_at_bang_id(self, children: list[Any]) -> NvidiaSassPredicate:
    """Transform a @!identifier predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassPredicate.
    """
    leading = _get_trivia(children[0])
    p = NvidiaSassPredicate(name=children[2].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_at_bang_reg(self, children: list[Any]) -> NvidiaSassPredicate:
    """Transform a @!register predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassPredicate.
    """
    leading = _get_trivia(children[0])
    p = NvidiaSassPredicate(name=children[2].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_at_id(self, children: list[Any]) -> NvidiaSassPredicate:
    """Transform a @identifier predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassPredicate.
    """
    leading = _get_trivia(children[0])
    p = NvidiaSassPredicate(name=children[1].value, negated=False, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_at_reg(self, children: list[Any]) -> NvidiaSassPredicate:
    """Transform a @register predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassPredicate.
    """
    leading = _get_trivia(children[0])
    p = NvidiaSassPredicate(name=children[1].value, negated=False, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_bang_id(self, children: list[Any]) -> NvidiaSassPredicate:
    """Transform a !identifier predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassPredicate.
    """
    leading = _get_trivia(children[0])
    p = NvidiaSassPredicate(name=children[1].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def pred_bang_reg(self, children: list[Any]) -> NvidiaSassPredicate:
    """Transform a !register predicate.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassPredicate.
    """
    leading = _get_trivia(children[0])
    p = NvidiaSassPredicate(name=children[1].value, negated=True, is_guard=False)
    p.leading_trivia = leading
    return p

  @v_args(inline=False)
  def immediate(self, children: list[Any]) -> NvidiaSassImmediate:
    """Transform an immediate value.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassImmediate.
    """
    leading = _get_trivia(children[0])
    val_str = children[0].value
    is_hex = "0x" in val_str.lower()
    if is_hex:
      val = cast(Any, int(val_str, 16))
    else:
      val2 = float(val_str) if "." in val_str else int(val_str, 10)
      val = cast(Any, val2)
    i = NvidiaSassImmediate(value=val, is_hex=is_hex)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def identifier(self, children: list[Any]) -> NvidiaSassLabel:
    """Transform an identifier operand.

    Args:
        children: Parsed children.

    Returns:
        The parsed NvidiaSassLabel.
    """
    leading = _get_trivia(children[0])
    lbl = NvidiaSassLabel(name=children[0].value)
    lbl.leading_trivia = leading
    return lbl


_CACHED_PARSER = None


class NvidiaSassParser:
  """Facade for parsing NVIDIA_SASS strings into CST modules."""

  def __init__(self, code: str) -> None:
    """Initialize the parser with the NVIDIA_SASS source code.

    Args:
        code: The raw NVIDIA_SASS string.
    """
    self.code = code
    global _CACHED_PARSER
    if _CACHED_PARSER is None:
      _CACHED_PARSER = Lark(GRAMMAR, parser="earley", lexer=NvidiaSassLexer)
    self.parser = _CACHED_PARSER
    self.transformer = NvidiaSassTransformer()

  def parse(self) -> NvidiaSassModule:
    """Parse the entire code block.

    Returns:
        NvidiaSassModule: The root CST node.
    """
    if not self.code.strip():
      return NvidiaSassModule()

    try:
      tree = self.parser.parse(self.code)
      return self.transformer.transform(tree)
    except Exception as e:
      raise ValueError(f"Unexpected token: {e}")
