"""Parser for the RDNA frontend.

Provides ``RdnaParser``, built on the table-driven :mod:`ml_switcheroo.core.compiler.isa.parser`
infrastructure to convert AMD RDNA / GCN assembly text into a Concrete Syntax Tree.
"""

import re
import typing

from lark import Transformer, Tree, v_args
from lark.lexer import Token

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaModule,
  RdnaNode,
  RdnaOperand,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.isa.parser import (
  IsaChild,
  IsaDialectSpec,
  IsaLexer,
  IsaParser,
  IsaToken,
  IsaTransformerBase,
)
from ml_switcheroo.core.compiler.isa.parser import (
  get_trivia as _get_trivia,
)
from ml_switcheroo.core.compiler.isa.parser import (
  tok_val as _tok_val,
)
from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.exceptions import SyntaxParseError

TOKEN_REGEX: tuple[tuple[str, str], ...] = (
  ("COMMENT", ";[^\x0a]*"),
  ("WS", "[\x20\x09\x0c\x0d\x0a]+"),
  ("HEX_NUMBER", r"0x[0-9a-fA-F]+"),
  ("NUMBER", r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"),
  ("STRING", '"(?:[^"\\x5c]|\\x5c.)*"'),
  ("REG_IDENTIFIER", r"[sv](?:\[\d+:\d+\]|\d+)"),
  ("MODIFIER", r"[a-zA-Z_][a-zA-Z0-9_\.]*(?::[^,\s\]]+|\([^)\s]*\))"),
  ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
  ("PUNCTUATION", r"[:,\[\]\+\-\(\)\.]"),
  ("MISMATCH", r"."),
)

PUNCTUATION_MAP: dict[str, str] = {
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

GRAMMAR: str = r"""
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

RDNA_SPEC = IsaDialectSpec(
  token_regex=TOKEN_REGEX,
  punctuation_map=PUNCTUATION_MAP,
  grammar=GRAMMAR,
  emit_eof_trivia=True,
  line_by_line=True,
)

tok_regex: str = RDNA_SPEC.compiled_regex
RdnaChild: typing.TypeAlias = IsaChild


class RdnaToken(IsaToken):
  """Custom RDNA token carrying leading trivia."""

  __slots__ = ()


class RdnaLexer(IsaLexer[RdnaToken]):
  """Table-driven Lark lexer for AMD RDNA preserving trivia.

  Attributes:
      SPEC: AMD RDNA lexical and grammar dialect specification.
  """

  SPEC: typing.ClassVar[IsaDialectSpec] = RDNA_SPEC

  def _make_token(self, kind: str, value: str, leading: list[Trivia]) -> RdnaToken:
    """Create an ``RdnaToken`` with attached ``leading`` trivia.

    Args:
        kind: Token terminal type name.
        value: Matched token substring.
        leading: Leading trivia items.

    Returns:
        RdnaToken: Constructed token instance.
    """
    tok = RdnaToken(kind, value)
    tok.leading_trivia = list(leading)
    return tok


class RdnaTransformer(
  IsaTransformerBase[
    RdnaNode,
    RdnaOperand,
    RdnaModule,
    RdnaComment,
    RdnaDirective,
    RdnaLabel,
    RdnaImmediate,
  ]
):
  """Transform parsed Lark trees into ``RdnaNode`` CST instances.

  Attributes:
      COMMENT_PREFIX_LEN: Comment prefix character count (``1`` for ``;``).
      STRIP_COMMENT_TEXT: Whether to strip leading/trailing whitespace from comment text.
  """

  COMMENT_PREFIX_LEN: typing.ClassVar[int] = 1
  STRIP_COMMENT_TEXT: typing.ClassVar[bool] = False

  def _is_statement_node(self, value: IsaChild) -> typing.TypeGuard[RdnaNode]:
    """Check whether ``value`` is an ``RdnaNode``.

    Args:
        value: Candidate parse child.

    Returns:
        bool: ``True`` if ``value`` is an ``RdnaNode``.
    """
    return isinstance(value, RdnaNode)

  def _is_operand_node(self, value: IsaChild) -> typing.TypeGuard[RdnaOperand]:
    """Check whether ``value`` is an ``RdnaOperand``.

    Args:
        value: Candidate parse child.

    Returns:
        bool: ``True`` if ``value`` is an ``RdnaOperand``.
    """
    return isinstance(value, RdnaOperand)

  def _create_module(self, statements: list[RdnaNode], leading: list[Trivia]) -> RdnaModule:
    """Construct ``RdnaModule`` with ``statements`` and ``leading`` trivia.

    Args:
        statements: Parsed statement nodes.
        leading: Module leading trivia.

    Returns:
        RdnaModule: Constructed module node.
    """
    return RdnaModule(statements=statements, leading_trivia=leading)

  def _create_comment(self, text: str, leading: list[Trivia]) -> RdnaComment:
    """Construct ``RdnaComment`` with ``text`` and ``leading`` trivia.

    Args:
        text: Comment body text.
        leading: Leading trivia items.

    Returns:
        RdnaComment: Constructed comment node.
    """
    return RdnaComment(text=text, leading_trivia=leading)

  def _create_directive(self, name: str, params: list[str], leading: list[Trivia]) -> RdnaDirective:
    """Construct ``RdnaDirective`` with ``name``, ``params``, and ``leading`` trivia.

    Args:
        name: Directive identifier.
        params: Directive parameter strings.
        leading: Leading trivia items.

    Returns:
        RdnaDirective: Constructed directive node.
    """
    return RdnaDirective(name=name, params=params, leading_trivia=leading)

  def _create_label(self, name: str, leading: list[Trivia]) -> RdnaLabel:
    """Construct ``RdnaLabel`` with ``name`` and ``leading`` trivia.

    Args:
        name: Label identifier.
        leading: Leading trivia items.

    Returns:
        RdnaLabel: Constructed label node.
    """
    return RdnaLabel(name=name, leading_trivia=leading)

  def _create_immediate(self, value: int | float, is_hex: bool, leading: list[Trivia]) -> RdnaImmediate:
    """Construct ``RdnaImmediate`` with ``value``, ``is_hex``, and ``leading`` trivia.

    Args:
        value: Numeric immediate value.
        is_hex: Hexadecimal formatting flag.
        leading: Leading trivia items.

    Returns:
        RdnaImmediate: Constructed immediate node.
    """
    return RdnaImmediate(value=value, is_hex=is_hex, leading_trivia=leading)

  @v_args(inline=False)
  def instruction(self, children: typing.Sequence[RdnaChild]) -> RdnaInstruction:
    """Transform an instruction line into ``RdnaInstruction``.

    Args:
        children: Opcode token and optional operand list.

    Returns:
        RdnaInstruction: Constructed instruction CST node.
    """
    opcode = _tok_val(children[0])
    leading = _get_trivia(children[0])
    operands: list[RdnaOperand] = []
    if len(children) > 1 and isinstance(children[1], list):
      operands = [op for op in children[1] if isinstance(op, RdnaOperand)]
    return RdnaInstruction(opcode=opcode, operands=operands, leading_trivia=leading)

  def _make_memory(self, children: typing.Sequence[RdnaChild], sign: int) -> RdnaMemory:
    """Construct an ``RdnaMemory`` operand with zero or signed immediate offset.

    Args:
        children: Bracket, base register, and optional sign/immediate children.
        sign: ``0`` for zero offset, ``1`` for positive offset, or ``-1`` for negative offset.

    Returns:
        RdnaMemory: Constructed memory operand CST node.
    """
    leading = _get_trivia(children[0])
    base = children[1]
    base_reg: RdnaSGPR | RdnaVGPR | str = base if isinstance(base, (RdnaSGPR, RdnaVGPR)) else str(base)
    if sign == 0:
      offset = 0
    else:
      imm = children[3]
      offset = sign * (int(imm.value) if isinstance(imm, RdnaImmediate) else 0)
    return RdnaMemory(base=base_reg, offset=offset, leading_trivia=leading)

  @v_args(inline=False)
  def mem_reg(self, children: typing.Sequence[RdnaChild]) -> RdnaMemory:
    """Transform bracketed register memory access ``[v0]``.

    Args:
        children: Bracket and register children.

    Returns:
        RdnaMemory: Memory operand with zero offset.
    """
    return self._make_memory(children, sign=0)

  @v_args(inline=False)
  def mem_reg_pos(self, children: typing.Sequence[RdnaChild]) -> RdnaMemory:
    """Transform bracketed register memory access with positive offset ``[v0 + 4]``.

    Args:
        children: Bracket, register, plus, and immediate children.

    Returns:
        RdnaMemory: Memory operand with positive offset.
    """
    return self._make_memory(children, sign=1)

  @v_args(inline=False)
  def mem_reg_neg(self, children: typing.Sequence[RdnaChild]) -> RdnaMemory:
    """Transform bracketed register memory access with negative offset ``[v0 - 4]``.

    Args:
        children: Bracket, register, minus, and immediate children.

    Returns:
        RdnaMemory: Memory operand with negative offset.
    """
    return self._make_memory(children, sign=-1)

  @v_args(inline=False)
  def register(self, children: typing.Sequence[RdnaChild]) -> RdnaSGPR | RdnaVGPR:
    """Transform an SGPR or VGPR register or register range token.

    Args:
        children: Register identifier token.

    Returns:
        RdnaSGPR | RdnaVGPR: Parsed scalar or vector register operand.

    Raises:
        SyntaxParseError: If the register token does not match SGPR/VGPR syntax.
    """
    leading = _get_trivia(children[0])
    val = _tok_val(children[0])

    match = re.match(r"^([sv])(?:\[(\d+):(\d+)\]|(\d+))$", val)
    if match is None:
      raise SyntaxParseError(f"Invalid RDNA register operand: '{val}'")

    is_sgpr = match.group(1) == "s"
    if match.group(2) and match.group(3):
      start = int(match.group(2))
      count = int(match.group(3)) - start + 1
    else:
      start = int(match.group(4))
      count = 1

    if is_sgpr:
      return RdnaSGPR(index=start, count=count, leading_trivia=leading)
    return RdnaVGPR(index=start, count=count, leading_trivia=leading)

  @v_args(inline=False)
  def imm_num(self, children: typing.Sequence[RdnaChild]) -> RdnaImmediate:
    """Transform an unsigned decimal or floating-point immediate.

    Args:
        children: Decimal number token.

    Returns:
        RdnaImmediate: Immediate operand CST node.
    """
    return self._parse_immediate(_tok_val(children[0]), leading=_get_trivia(children[0]), is_hex=False, sign=1)

  @v_args(inline=False)
  def imm_hex(self, children: typing.Sequence[RdnaChild]) -> RdnaImmediate:
    """Transform an unsigned hexadecimal immediate.

    Args:
        children: Hexadecimal number token.

    Returns:
        RdnaImmediate: Hexadecimal immediate operand CST node.
    """
    return self._parse_immediate(_tok_val(children[0]), leading=_get_trivia(children[0]), is_hex=True, sign=1)

  @v_args(inline=False)
  def neg_num(self, children: typing.Sequence[RdnaChild]) -> RdnaImmediate:
    """Transform a negative decimal or floating-point immediate.

    Args:
        children: Minus token and decimal number token.

    Returns:
        RdnaImmediate: Negative immediate operand CST node.
    """
    return self._parse_immediate(_tok_val(children[1]), leading=_get_trivia(children[0]), is_hex=False, sign=-1)

  @v_args(inline=False)
  def neg_hex(self, children: typing.Sequence[RdnaChild]) -> RdnaImmediate:
    """Transform a negative hexadecimal immediate.

    Args:
        children: Minus token and hexadecimal number token.

    Returns:
        RdnaImmediate: Negative hexadecimal immediate operand CST node.
    """
    return self._parse_immediate(_tok_val(children[1]), leading=_get_trivia(children[0]), is_hex=True, sign=-1)

  @v_args(inline=False)
  def pos_num(self, children: typing.Sequence[RdnaChild]) -> RdnaImmediate:
    """Transform an explicit ``+`` decimal or floating-point immediate.

    Args:
        children: Plus token and decimal number token.

    Returns:
        RdnaImmediate: Positive immediate operand CST node.
    """
    return self._parse_immediate(_tok_val(children[1]), leading=_get_trivia(children[0]), is_hex=False, sign=1)

  @v_args(inline=False)
  def pos_hex(self, children: typing.Sequence[RdnaChild]) -> RdnaImmediate:
    """Transform an explicit ``+`` hexadecimal immediate.

    Args:
        children: Plus token and hexadecimal number token.

    Returns:
        RdnaImmediate: Positive hexadecimal immediate operand CST node.
    """
    return self._parse_immediate(_tok_val(children[1]), leading=_get_trivia(children[0]), is_hex=True, sign=1)

  @v_args(inline=False)
  def modifier(self, children: typing.Sequence[RdnaChild]) -> RdnaModifier:
    """Transform an instruction modifier token (e.g., ``offset:4``).

    Args:
        children: Modifier token.

    Returns:
        RdnaModifier: Modifier operand CST node.
    """
    return RdnaModifier(name=_tok_val(children[0]), leading_trivia=_get_trivia(children[0]))

  @v_args(inline=False)
  def ident_or_modifier(self, children: typing.Sequence[RdnaChild]) -> RdnaModifier | RdnaLabelRef:
    """Transform an identifier operand into either ``RdnaModifier`` or ``RdnaLabelRef``.

    Args:
        children: Identifier token.

    Returns:
        RdnaModifier | RdnaLabelRef: Modifier node for ``off``/``glc``/``slc``, else label reference.
    """
    leading = _get_trivia(children[0])
    val = _tok_val(children[0])
    if val in ("off", "glc", "slc"):
      return RdnaModifier(name=val, leading_trivia=leading)
    return RdnaLabelRef(name=val, leading_trivia=leading)


class RdnaParser(IsaParser[RdnaModule, RdnaNode]):
  """Facade for parsing AMD RDNA source text into ``RdnaModule`` CSTs.

  Attributes:
      SPEC: AMD RDNA lexical and grammar dialect specification.
      LEXER_CLS: Lexer class bound to Lark Earley construction.
  """

  SPEC: typing.ClassVar[IsaDialectSpec] = RDNA_SPEC
  LEXER_CLS: typing.ClassVar[type[RdnaLexer]] = RdnaLexer

  def _build_transformer(self) -> Transformer[Token, RdnaNode]:
    """Instantiate ``RdnaTransformer``.

    Returns:
        Transformer[Token, RdnaNode]: Dialect transformer instance.
    """
    return RdnaTransformer()

  def _empty_module(self, statements: list[RdnaNode] | None = None) -> RdnaModule:
    """Construct an ``RdnaModule``.

    Args:
        statements: Optional parsed statements.

    Returns:
        RdnaModule: Constructed module node.
    """
    return RdnaModule(statements=statements)

  def _validate_module(self, node: RdnaNode | Tree[Token]) -> tuple[RdnaModule, list[RdnaNode]]:
    """Validate that Lark reduction produced an ``RdnaModule``.

    Args:
        node: Root object produced by ``RdnaTransformer``.

    Returns:
        tuple[RdnaModule, list[RdnaNode]]: Validated module and its statement list.

    Raises:
        SyntaxParseError: If ``node`` is not an ``RdnaModule``.
    """
    if not isinstance(node, RdnaModule):
      raise SyntaxParseError(f"Expected RdnaModule root, got {type(node).__name__}")
    return node, list(node.statements)


__all__ = [
  "RdnaComment",
  "RdnaDirective",
  "RdnaImmediate",
  "RdnaInstruction",
  "RdnaLabel",
  "RdnaLabelRef",
  "RdnaLexer",
  "RdnaMemory",
  "RdnaModifier",
  "RdnaModule",
  "RdnaNode",
  "RdnaOperand",
  "RdnaParser",
  "RdnaSGPR",
  "RdnaToken",
  "RdnaTransformer",
  "RdnaVGPR",
  "Trivia",
  "_get_trivia",
]
