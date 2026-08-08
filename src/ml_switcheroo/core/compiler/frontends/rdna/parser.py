"""Parser for the RDNA frontend.

This module provides the `RdnaParser`, a pure Python Lark-based parser
that converts a stream of characters into a Concrete Syntax Tree defined in `cst.py`.
"""

import re
from typing import List, Any, Union, cast

from lark import Lark, Transformer, v_args
from lark.lexer import Lexer, Token

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaModifier,
  RdnaOperand,
  RdnaSGPR,
  RdnaVGPR,
  RdnaMemory,
  RdnaModule,
  RdnaNode,
)
from ml_switcheroo.core.cst.base import Trivia

TOKEN_REGEX = [
  ("COMMENT", r";[^\n]*"),
  ("WS", r"[ \t\f\r\n]+"),
  ("HEX_NUMBER", r"0x[0-9a-fA-F]+"),
  ("NUMBER", r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"),
  ("STRING", r'"(?:[^"\\]|\\.)*"'),
  ("REG_IDENTIFIER", r"[sv](?:\[\d+:\d+\]|\d+)"),
  ("MODIFIER", r"[a-zA-Z_][a-zA-Z0-9_\.]*(?::[^,\s\]]+|\([^)\s]*\))"),
  ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
  ("PUNCTUATION", r"[:,\[\]\+\-\(\)\.]"),
  ("MISMATCH", r"."),
]

tok_regex = "|".join("(?P<%s>%s)" % pair for pair in TOKEN_REGEX)


class RdnaToken(Token):
  """Custom token that carries its leading trivia."""

  __slots__ = ("leading_trivia",)

  leading_trivia: List[Trivia]


class RdnaLexer(Lexer):
  """Custom Lexer preserving trivia and matching RDNA tokens."""

  def __init__(self, lexer_conf: Any) -> None:
    """Initializes the custom RDNA lexer.

    Args:
        lexer_conf: The configuration settings for the Lark lexer.
    """
    self.lexer_conf = lexer_conf

  def lex(self, data: str) -> Any:  # type: ignore[override]
    """Tokenize the input string and attach trivia.

    Args:
        data: The source string to tokenize.

    Yields:
        RdnaToken: Tokens.

    Raises:
        ValueError: Value error.
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
        t = RdnaToken("COMMENT", val)
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
            "(": "LPAREN",
            ")": "RPAREN",
          }
          assert val is not None
          kind = punct_map[val]

        assert kind is not None
        assert val is not None
        t2 = RdnaToken(kind, val)
        t2.leading_trivia = list(leading)
        leading.clear()
        yield t2
    if leading:
      t3 = RdnaToken("EOF_TRIVIA", "")
      t3.leading_trivia = list(leading)
      yield t3


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
  return []


GRAMMAR = r"""
    ?start: module
    module: statement* [EOF_TRIVIA]

    ?statement: instruction | directive | label | comment_stmt

    comment_stmt: COMMENT

    directive: DOT IDENTIFIER [directive_params]
    directive_params: param_list
    param_list: directive_param ([COMMA] directive_param)*
    ?directive_param: IDENTIFIER | NUMBER | HEX_NUMBER | MINUS NUMBER | MINUS HEX_NUMBER | PLUS NUMBER | PLUS HEX_NUMBER | MODIFIER | STRING

    label: IDENTIFIER COLON

    instruction: IDENTIFIER [operands]

    operands: operand (COMMA operand)*

    ?operand: memory | register | immediate | MODIFIER -> modifier | IDENTIFIER -> ident_or_modifier

    memory: LBRACK register RBRACK -> mem_reg
          | LBRACK register PLUS immediate RBRACK -> mem_reg_pos
          | LBRACK register MINUS immediate RBRACK -> mem_reg_neg

    register: REG_IDENTIFIER

    immediate: NUMBER -> imm_num
             | HEX_NUMBER -> imm_hex
             | MINUS NUMBER -> neg_num
             | MINUS HEX_NUMBER -> neg_hex
             | PLUS NUMBER -> pos_num
             | PLUS HEX_NUMBER -> pos_hex

    SEMI: ";"
    COLON: ":"
    COMMA: ","
    DOT: "."
    LBRACK: "["
    RBRACK: "]"
    LPAREN: "("
    RPAREN: ")"
    PLUS: "+"
    MINUS: "-"

    IDENTIFIER: /.+/
    REG_IDENTIFIER: /.+/
    NUMBER: /.+/
    HEX_NUMBER: /.+/
    STRING: /.+/
    COMMENT: /.+/
    MODIFIER: /.+/
    EOF_TRIVIA: /.*/
"""


class RdnaTransformer(Transformer[Any, Any]):
  """Transforms parsed AST nodes into RdnaNode classes.

  This class traverses the Lark parse tree and constructs the equivalent Concrete
  Syntax Tree (CST) representation using specialized RDNA node types.
  """

  @v_args(inline=False)
  def module(self, children: List[Any]) -> RdnaModule:
    """Transform the top-level module rule.

    Args:
        children: A list of statement nodes representing the parsed code.

    Returns:
        RdnaModule: The completed module node containing the list of statements.
    """
    leading: List[Trivia] = []
    stmts: List[RdnaNode] = []
    for c in children:
      if c is not None and not isinstance(c, Token):
        stmts.append(c)
      elif isinstance(c, Token) and c.type == "EOF_TRIVIA":
        if stmts:
          stmts[-1].trailing_trivia.extend(_get_trivia(c))

    mod = RdnaModule(statements=stmts)
    mod.leading_trivia = leading
    return mod

  @v_args(inline=False)
  def comment_stmt(self, children: List[Any]) -> RdnaComment:
    """Transform a comment.

    Args:
        children: A list containing the comment token.

    Returns:
        RdnaComment: The transformed comment node.
    """
    c = RdnaComment(text=children[0].value[1:])
    c.leading_trivia = _get_trivia(children[0])
    return c

  @v_args(inline=False)
  def directive(self, children: List[Any]) -> RdnaDirective:
    """Transform a directive.

    Args:
        children: The list of parsed elements comprising the directive name and parameters.

    Returns:
        RdnaDirective: The parsed directive node with its name and parameters.
    """
    name = children[1].value
    params: List[str] = []
    if len(children) > 2 and children[2] is not None:
      param_list = children[2].children[0] if getattr(children[2], "data", None) == "directive_params" else children[2]
      if isinstance(param_list, list):
        for p in param_list:
          if getattr(p, "type", None) == "COMMA":
            continue
          if hasattr(p, "children"):
            params.append("".join(getattr(c, "value", str(c)) for c in p.children))
          else:
            params.append(str(p))
      else:
        params.append(str(param_list))
    d = RdnaDirective(name=name, params=params)
    d.leading_trivia = _get_trivia(children[0])
    return d

  @v_args(inline=False)
  def param_list(self, children: List[Any]) -> List[Any]:
    """Transform a parameter list.

    Args:
        children: The parameters parsed in the list.

    Returns:
        List[Any]: The simplified list of parameter objects.
    """
    return children

  @v_args(inline=False)
  def label(self, children: List[Any]) -> RdnaLabel:
    """Transform a label.

    Args:
        children: The list containing the label's identifier token.

    Returns:
        RdnaLabel: The constructed label node.
    """
    lbl = RdnaLabel(name=children[0].value)
    lbl.leading_trivia = _get_trivia(children[0])
    return lbl

  @v_args(inline=False)
  def instruction(self, children: List[Any]) -> RdnaInstruction:
    """Transform an instruction.

    Args:
        children: The parsed opcode and operand list.

    Returns:
        RdnaInstruction: The constructed instruction node.
    """
    opcode = children[0].value
    operands: List[RdnaOperand] = []
    leading = _get_trivia(children[0])

    if len(children) > 1 and children[1] is not None:
      operands = children[1]

    i = RdnaInstruction(opcode=opcode, operands=operands)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def operands(self, children: List[Any]) -> List[RdnaOperand]:
    """Transform an operands list.

    Args:
        children: The operands associated with an instruction.

    Returns:
        List[RdnaOperand]: Filtered and structured instruction operand nodes.
    """
    return [c for c in children if isinstance(c, RdnaOperand)]

  @v_args(inline=False)
  def mem_reg(self, children: List[Any]) -> RdnaMemory:
    """Transform memory access.

    Args:
        children: The list of elements, specifically the base register.

    Returns:
        RdnaMemory: A memory operand representing the register-based access with zero offset.
    """
    leading = _get_trivia(children[0])
    m = RdnaMemory(base=children[1], offset=0)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg_pos(self, children: List[Any]) -> RdnaMemory:
    """Transform memory access with positive offset.

    Args:
        children: The list containing the base register and the offset immediate.

    Returns:
        RdnaMemory: A memory operand representing the register-based access with positive offset.
    """
    leading = _get_trivia(children[0])
    imm = children[3]
    val = imm.value if hasattr(imm, "value") else getattr(imm, "value", 0)
    m = RdnaMemory(base=children[1], offset=int(val))
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def mem_reg_neg(self, children: List[Any]) -> RdnaMemory:
    """Transform memory access with negative offset.

    Args:
        children: The list containing the base register and the offset immediate.

    Returns:
        RdnaMemory: A memory operand representing the register-based access with negative offset.
    """
    leading = _get_trivia(children[0])
    imm = children[3]
    val = imm.value if hasattr(imm, "value") else getattr(imm, "value", 0)
    m = RdnaMemory(base=children[1], offset=-int(val))
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def register(self, children: List[Any]) -> Union[RdnaSGPR, RdnaVGPR]:
    """Transform a register.

    Args:
        children: The register identifier token.

    Returns:
        Union[RdnaSGPR, RdnaVGPR]: The parsed register node (SGPR or VGPR) with its range.
    """
    leading = _get_trivia(children[0])
    val = children[0].value

    match = re.match(r"^([sv])(?:\[(\d+):(\d+)\]|(\d+))$", val)
    assert match is not None

    is_sgpr = match.group(1) == "s"
    if match.group(2) and match.group(3):
      start = int(match.group(2))
      count = int(match.group(3)) - start + 1
    else:
      start = int(match.group(4))
      count = 1

    reg: Union[RdnaSGPR, RdnaVGPR] = RdnaSGPR(index=start, count=count) if is_sgpr else RdnaVGPR(index=start, count=count)
    reg.leading_trivia = leading
    return reg

  @v_args(inline=False)
  def imm_num(self, children: List[Any]) -> RdnaImmediate:
    """Transform number.

    Args:
        children: The decimal number token.

    Returns:
        RdnaImmediate: The immediate operand holding the parsed numeric value.
    """
    leading = _get_trivia(children[0])
    val_str = children[0].value
    val = float(val_str) if "." in val_str else int(val_str, 10)
    i = RdnaImmediate(value=val, is_hex=False)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def imm_hex(self, children: List[Any]) -> RdnaImmediate:
    """Transform hex.

    Args:
        children: The hexadecimal number token.

    Returns:
        RdnaImmediate: The immediate operand holding the parsed hexadecimal value.
    """
    leading = _get_trivia(children[0])
    val = int(children[0].value, 16)
    i = RdnaImmediate(value=val, is_hex=True)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def neg_num(self, children: List[Any]) -> RdnaImmediate:
    """Transform negative number.

    Args:
        children: The minus sign token and the decimal number token.

    Returns:
        RdnaImmediate: The negative immediate operand holding the parsed numeric value.
    """
    leading = _get_trivia(children[0])
    val_str = children[1].value
    val = float(val_str) if "." in val_str else int(val_str, 10)
    i = RdnaImmediate(value=-val, is_hex=False)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def neg_hex(self, children: List[Any]) -> RdnaImmediate:
    """Transform negative hex.

    Args:
        children: The minus sign token and the hexadecimal number token.

    Returns:
        RdnaImmediate: The negative immediate operand holding the parsed hexadecimal value.
    """
    leading = _get_trivia(children[0])
    val = int(children[1].value, 16)
    i = RdnaImmediate(value=-val, is_hex=True)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def pos_num(self, children: List[Any]) -> RdnaImmediate:
    """Transform positive number.

    Args:
        children: The plus sign token and the decimal number token.

    Returns:
        RdnaImmediate: The positive immediate operand holding the parsed numeric value.
    """
    leading = _get_trivia(children[0])
    val_str = children[1].value
    val = float(val_str) if "." in val_str else int(val_str, 10)
    i = RdnaImmediate(value=val, is_hex=False)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def pos_hex(self, children: List[Any]) -> RdnaImmediate:
    """Transform positive hex.

    Args:
        children: The plus sign token and the hexadecimal number token.

    Returns:
        RdnaImmediate: The positive immediate operand holding the parsed hexadecimal value.
    """
    leading = _get_trivia(children[0])
    val = int(children[1].value, 16)
    i = RdnaImmediate(value=val, is_hex=True)
    i.leading_trivia = leading
    return i

  @v_args(inline=False)
  def modifier(self, children: List[Any]) -> RdnaModifier:
    """Transform a modifier.

    Args:
        children: The modifier token.

    Returns:
        RdnaModifier: The constructed modifier node.
    """
    leading = _get_trivia(children[0])
    m = RdnaModifier(name=children[0].value)
    m.leading_trivia = leading
    return m

  @v_args(inline=False)
  def ident_or_modifier(self, children: List[Any]) -> Union[RdnaModifier, RdnaLabelRef]:
    """Transform an identifier which could be a modifier.

    Args:
        children: The identifier token.

    Returns:
        Union[RdnaModifier, RdnaLabelRef]: Either a modifier or a label reference.
    """
    leading = _get_trivia(children[0])
    val = children[0].value
    if val in ("off", "glc", "slc"):
      res: Union[RdnaModifier, RdnaLabelRef] = RdnaModifier(name=val)
    else:
      res = RdnaLabelRef(name=val)
    res.leading_trivia = leading
    return res


class RdnaParser:
  """Facade for parsing RDNA strings into CST modules.

  This class acts as the main entry point to parse RDNA source files and convert
  them into Concrete Syntax Trees (CST).
  """

  def __init__(self, code: str) -> None:
    """Initialize the parser with the RDNA source code.

    Args:
        code: The raw RDNA string.
    """
    self.code = code
    self.parser = Lark(GRAMMAR, parser="earley", lexer=RdnaLexer)
    self.transformer = RdnaTransformer()

  def parse(self) -> RdnaModule:
    """Parses the entire code block.

    Returns:
        RdnaModule: The root CST node.

    Raises:
        ValueError: Parse error.
    """
    if not self.code.strip():
      return RdnaModule()

    statements: List[RdnaNode] = []

    # We parse line-by-line to exactly mirror original parser behaviour
    for line in self.code.splitlines(keepends=True):
      if not line.strip():
        continue
      try:
        tree = self.parser.parse(line)
        mod = cast(RdnaModule, self.transformer.transform(tree))
        statements.extend(mod.statements)
      except Exception as e:
        raise ValueError(f"Unexpected token: {e}")

    return RdnaModule(statements=statements)
