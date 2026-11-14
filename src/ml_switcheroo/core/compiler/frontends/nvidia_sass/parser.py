"""NVIDIA_SASS Parser Implementation.

Provides ``NvidiaSassParser``, built on the table-driven :mod:`ml_switcheroo.core.compiler.isa.parser`
infrastructure to convert NVIDIA SASS assembly text into a Concrete Syntax Tree.
"""

import typing

from lark import Transformer, Tree, v_args
from lark.lexer import Token

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassModule,
  NvidiaSassNode,
  NvidiaSassOperand,
  NvidiaSassPredicate,
  NvidiaSassRegister,
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
  ("COMMENT", "//[^\x0a]*"),
  ("WS", "[\x20\x09\x0c\x0d\x0a]+"),
  ("HEX_NUMBER", r"-?0x[0-9a-fA-F]+"),
  ("NUMBER", r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"),
  ("STRING", '"(?:[^"\\x5c]|\\x5c.)*"'),
  ("REG_IDENTIFIER", r"[-|]*[RU]?[RZS]\d+\|?|PT|RZ|RZS"),
  ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_\.]*"),
  ("PUNCTUATION", r"[;:,\.\[\]\+\-!@]"),
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
  "!": "BANG",
  "@": "AT",
}

GRAMMAR: str = r"""
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

NVIDIA_SASS_SPEC = IsaDialectSpec(
  token_regex=TOKEN_REGEX,
  punctuation_map=PUNCTUATION_MAP,
  grammar=GRAMMAR,
  emit_eof_trivia=False,
  line_by_line=False,
)

tok_regex: str = NVIDIA_SASS_SPEC.compiled_regex
SassChild: typing.TypeAlias = IsaChild


class NvidiaSassToken(IsaToken):
  """Custom NVIDIA SASS token carrying leading trivia."""

  __slots__ = ()


class NvidiaSassLexer(IsaLexer[NvidiaSassToken]):
  """Table-driven Lark lexer for NVIDIA SASS preserving trivia.

  Attributes:
      SPEC: NVIDIA SASS lexical and grammar dialect specification.
  """

  SPEC: typing.ClassVar[IsaDialectSpec] = NVIDIA_SASS_SPEC

  def _make_token(self, kind: str, value: str, leading: list[Trivia]) -> NvidiaSassToken:
    """Create a ``NvidiaSassToken`` with attached ``leading`` trivia.

    Args:
        kind: Token terminal type name.
        value: Matched token substring.
        leading: Leading trivia items.

    Returns:
        NvidiaSassToken: Constructed token instance.
    """
    tok = NvidiaSassToken(kind, value)
    tok.leading_trivia = list(leading)
    return tok


class NvidiaSassTransformer(
  IsaTransformerBase[
    NvidiaSassNode,
    NvidiaSassOperand,
    NvidiaSassModule,
    NvidiaSassComment,
    NvidiaSassDirective,
    NvidiaSassLabel,
    NvidiaSassImmediate,
  ]
):
  """Transform parsed Lark trees into ``NvidiaSassNode`` CST instances.

  Attributes:
      COMMENT_PREFIX_LEN: Comment prefix character count (``2`` for ``//``).
      STRIP_COMMENT_TEXT: Whether to strip leading/trailing whitespace from comment text.
      HOIST_FIRST_STMT_TRIVIA: Whether to hoist first-statement leading trivia onto the module.
  """

  COMMENT_PREFIX_LEN: typing.ClassVar[int] = 2
  STRIP_COMMENT_TEXT: typing.ClassVar[bool] = True
  HOIST_FIRST_STMT_TRIVIA: typing.ClassVar[bool] = True

  def _is_statement_node(self, value: IsaChild) -> typing.TypeGuard[NvidiaSassNode]:
    """Check whether ``value`` is a ``NvidiaSassNode``.

    Args:
        value: Candidate parse child.

    Returns:
        bool: ``True`` if ``value`` is a ``NvidiaSassNode``.
    """
    return isinstance(value, NvidiaSassNode)

  def _is_operand_node(self, value: IsaChild) -> typing.TypeGuard[NvidiaSassOperand]:
    """Check whether ``value`` is a ``NvidiaSassOperand``.

    Args:
        value: Candidate parse child.

    Returns:
        bool: ``True`` if ``value`` is a ``NvidiaSassOperand``.
    """
    return isinstance(value, NvidiaSassOperand)

  def _create_module(self, statements: list[NvidiaSassNode], leading: list[Trivia]) -> NvidiaSassModule:
    """Construct ``NvidiaSassModule`` with ``statements`` and ``leading`` trivia.

    Args:
        statements: Parsed statement nodes.
        leading: Module leading trivia.

    Returns:
        NvidiaSassModule: Constructed module node.
    """
    return NvidiaSassModule(statements=statements, leading_trivia=leading)

  def _create_comment(self, text: str, leading: list[Trivia]) -> NvidiaSassComment:
    """Construct ``NvidiaSassComment`` with ``text`` and ``leading`` trivia.

    Args:
        text: Stripped comment text.
        leading: Leading trivia items.

    Returns:
        NvidiaSassComment: Constructed comment node.
    """
    return NvidiaSassComment(text=text, leading_trivia=leading)

  def _create_directive(self, name: str, params: list[str], leading: list[Trivia]) -> NvidiaSassDirective:
    """Construct ``NvidiaSassDirective`` with ``name``, ``params``, and ``leading`` trivia.

    Args:
        name: Directive identifier.
        params: Directive parameter strings.
        leading: Leading trivia items.

    Returns:
        NvidiaSassDirective: Constructed directive node.
    """
    return NvidiaSassDirective(name=name, params=params, leading_trivia=leading)

  def _create_label(self, name: str, leading: list[Trivia]) -> NvidiaSassLabel:
    """Construct ``NvidiaSassLabel`` with ``name`` and ``leading`` trivia.

    Args:
        name: Label identifier.
        leading: Leading trivia items.

    Returns:
        NvidiaSassLabel: Constructed label node.
    """
    return NvidiaSassLabel(name=name, leading_trivia=leading)

  def _create_immediate(self, value: int | float, is_hex: bool, leading: list[Trivia]) -> NvidiaSassImmediate:
    """Construct ``NvidiaSassImmediate`` with ``value``, ``is_hex``, and ``leading`` trivia.

    Args:
        value: Numeric immediate value.
        is_hex: Hexadecimal formatting flag.
        leading: Leading trivia items.

    Returns:
        NvidiaSassImmediate: Constructed immediate node.
    """
    return NvidiaSassImmediate(value=value, is_hex=is_hex, leading_trivia=leading)

  @v_args(inline=False)
  def empty_statement(self, children: typing.Sequence[SassChild]) -> None:
    """Discard standalone semicolon empty statements.

    Args:
        children: Semicolon token list.

    Returns:
        None: Empty statements produce no CST node.
    """
    return

  @v_args(inline=False)
  def at_string(self, children: list[SassChild]) -> list[SassChild]:
    """Pass through ``@"..."`` directive parameter tokens.

    Args:
        children: ``AT`` and ``STRING`` tokens.

    Returns:
        list[SassChild]: Unmodified token list.
    """
    return children

  @v_args(inline=False)
  def instruction(self, children: typing.Sequence[SassChild]) -> NvidiaSassInstruction:
    """Transform an instruction line into a ``NvidiaSassInstruction``.

    Args:
        children: Optional predicate guard, opcode token, operand list, and semicolon.

    Returns:
        NvidiaSassInstruction: Constructed instruction CST node.
    """
    predicate: NvidiaSassPredicate | None = None
    opcode = ""
    operands: list[NvidiaSassOperand] = []

    leading: list[Trivia] = []
    for c in children:
      if c is not None:
        leading = _get_trivia(c)
        break

    for c in children:
      if isinstance(c, NvidiaSassPredicate):
        predicate = c
      elif isinstance(c, Token) and c.type == "IDENTIFIER":
        opcode = c.value
      elif isinstance(c, list):
        operands = [op for op in c if isinstance(op, NvidiaSassOperand)]

    return NvidiaSassInstruction(opcode=opcode, operands=operands, predicate=predicate, leading_trivia=leading)

  @v_args(inline=False)
  def predicate(self, children: typing.Sequence[SassChild]) -> NvidiaSassPredicate:
    """Transform an ``@P0`` / ``@!P0`` instruction guard predicate.

    Args:
        children: ``AT``, optional ``BANG``, and predicate identifier tokens.

    Returns:
        NvidiaSassPredicate: Guard predicate CST node.
    """
    negated = False
    name = ""
    leading = _get_trivia(children[0])
    for c in children:
      if isinstance(c, Token) and c.type == "BANG":
        negated = True
      elif isinstance(c, Token) and c.type in ("IDENTIFIER", "REG_IDENTIFIER"):
        name = str(c.value)
    return NvidiaSassPredicate(name=name, negated=negated, is_guard=True, leading_trivia=leading)

  @v_args(inline=False)
  def mem_bank(self, children: typing.Sequence[SassChild]) -> NvidiaSassMemory:
    """Transform constant bank memory access ``c[0x0][0x4]``.

    Args:
        children: Bank identifier and hex index tokens.

    Returns:
        NvidiaSassMemory: Constant bank memory operand CST node.
    """
    leading = _get_trivia(children[0])
    bank = int(_tok_val(children[2]), 16)
    offset = int(_tok_val(children[5]), 16)
    base_name = _tok_val(children[0])
    return NvidiaSassMemory(base=f"{base_name}[{hex(bank)}]", offset=offset, leading_trivia=leading)

  @v_args(inline=False)
  def mem_bank_single(self, children: typing.Sequence[SassChild]) -> NvidiaSassMemory:
    """Transform single-bracket bank memory access ``c[0x0]``.

    Args:
        children: Bank identifier and hex index tokens.

    Returns:
        NvidiaSassMemory: Single-bracket memory operand CST node.
    """
    leading = _get_trivia(children[0])
    bank = int(_tok_val(children[2]), 16)
    base_name = _tok_val(children[0])
    return NvidiaSassMemory(base=f"{base_name}[{hex(bank)}]", offset=None, leading_trivia=leading)

  @v_args(inline=False)
  def mem_reg(self, children: typing.Sequence[SassChild]) -> NvidiaSassMemory:
    """Transform register-indirect memory access ``[R0]``.

    Args:
        children: Bracket and base register children.

    Returns:
        NvidiaSassMemory: Register memory operand CST node.
    """
    leading = _get_trivia(children[0])
    base = children[1]
    base_reg: str | NvidiaSassRegister = base if isinstance(base, NvidiaSassRegister) else str(base)
    return NvidiaSassMemory(base=base_reg, offset=None, leading_trivia=leading)

  def _mem_reg_with_signed_offset(self, children: typing.Sequence[SassChild], sign: int) -> NvidiaSassMemory:
    """Construct a register-indirect memory operand with signed byte offset.

    Args:
        children: Bracket, register, sign, and immediate offset children.
        sign: ``1`` for positive offset or ``-1`` for negative offset.

    Returns:
        NvidiaSassMemory: Constructed memory operand CST node.
    """
    leading = _get_trivia(children[0])
    c3 = children[3]
    offset_tok = c3.children[0] if isinstance(c3, Tree) and c3.children else c3
    raw_offset = _tok_val(offset_tok)
    offset = sign * int(raw_offset, 16 if "0x" in raw_offset.lower() else 10)
    base = children[1]
    base_reg: str | NvidiaSassRegister = base if isinstance(base, NvidiaSassRegister) else str(base)
    return NvidiaSassMemory(base=base_reg, offset=offset, leading_trivia=leading)

  @v_args(inline=False)
  def mem_reg_offset(self, children: typing.Sequence[SassChild]) -> NvidiaSassMemory:
    """Transform register memory access with positive offset ``[R0 + 0x4]``.

    Args:
        children: Bracket, register, plus, and offset children.

    Returns:
        NvidiaSassMemory: Memory operand CST node.
    """
    return self._mem_reg_with_signed_offset(children, sign=1)

  @v_args(inline=False)
  def mem_reg_neg_offset(self, children: typing.Sequence[SassChild]) -> NvidiaSassMemory:
    """Transform register memory access with negative offset ``[R0 - 0x4]``.

    Args:
        children: Bracket, register, minus, and offset children.

    Returns:
        NvidiaSassMemory: Memory operand CST node.
    """
    return self._mem_reg_with_signed_offset(children, sign=-1)

  @v_args(inline=False)
  def register(self, children: typing.Sequence[SassChild]) -> NvidiaSassRegister:
    """Transform a SASS register token (supporting ``-`` negation and ``|...|`` abs modifiers).

    Args:
        children: Register identifier token list.

    Returns:
        NvidiaSassRegister: Constructed register operand CST node.
    """
    leading = _get_trivia(children[0])
    val = _tok_val(children[0])
    negated = val.startswith(("-", "|-"))
    absolute = "|" in val
    name = val.replace("-", "").replace("|", "")
    return NvidiaSassRegister(name=name, negated=negated, absolute=absolute, leading_trivia=leading)

  def _operand_predicate(
    self,
    children: typing.Sequence[SassChild],
    *,
    name_idx: int,
    negated: bool,
  ) -> NvidiaSassPredicate:
    """Construct a non-guard ``NvidiaSassPredicate`` operand.

    Args:
        children: Matched predicate operand tokens.
        name_idx: Index of the identifier/register token in ``children``.
        negated: Whether ``!`` negation was matched.

    Returns:
        NvidiaSassPredicate: Predicate operand CST node.
    """
    return NvidiaSassPredicate(
      name=_tok_val(children[name_idx]),
      negated=negated,
      is_guard=False,
      leading_trivia=_get_trivia(children[0]),
    )

  @v_args(inline=False)
  def pred_at_bang_id(self, children: typing.Sequence[SassChild]) -> NvidiaSassPredicate:
    """Transform an ``@!identifier`` predicate operand.

    Args:
        children: Matched tokens.

    Returns:
        NvidiaSassPredicate: Negated predicate operand.
    """
    return self._operand_predicate(children, name_idx=2, negated=True)

  @v_args(inline=False)
  def pred_at_bang_reg(self, children: typing.Sequence[SassChild]) -> NvidiaSassPredicate:
    """Transform an ``@!register`` predicate operand.

    Args:
        children: Matched tokens.

    Returns:
        NvidiaSassPredicate: Negated predicate operand.
    """
    return self._operand_predicate(children, name_idx=2, negated=True)

  @v_args(inline=False)
  def pred_at_id(self, children: typing.Sequence[SassChild]) -> NvidiaSassPredicate:
    """Transform an ``@identifier`` predicate operand.

    Args:
        children: Matched tokens.

    Returns:
        NvidiaSassPredicate: Predicate operand.
    """
    return self._operand_predicate(children, name_idx=1, negated=False)

  @v_args(inline=False)
  def pred_at_reg(self, children: typing.Sequence[SassChild]) -> NvidiaSassPredicate:
    """Transform an ``@register`` predicate operand.

    Args:
        children: Matched tokens.

    Returns:
        NvidiaSassPredicate: Predicate operand.
    """
    return self._operand_predicate(children, name_idx=1, negated=False)

  @v_args(inline=False)
  def pred_bang_id(self, children: typing.Sequence[SassChild]) -> NvidiaSassPredicate:
    """Transform a ``!identifier`` predicate operand.

    Args:
        children: Matched tokens.

    Returns:
        NvidiaSassPredicate: Negated predicate operand.
    """
    return self._operand_predicate(children, name_idx=1, negated=True)

  @v_args(inline=False)
  def pred_bang_reg(self, children: typing.Sequence[SassChild]) -> NvidiaSassPredicate:
    """Transform a ``!register`` predicate operand.

    Args:
        children: Matched tokens.

    Returns:
        NvidiaSassPredicate: Negated predicate operand.
    """
    return self._operand_predicate(children, name_idx=1, negated=True)

  @v_args(inline=False)
  def immediate(self, children: typing.Sequence[SassChild]) -> NvidiaSassImmediate:
    """Transform a decimal, float, or hexadecimal immediate literal.

    Args:
        children: Immediate literal token.

    Returns:
        NvidiaSassImmediate: Constructed immediate operand CST node.
    """
    return self._parse_immediate(_tok_val(children[0]), leading=_get_trivia(children[0]))

  @v_args(inline=False)
  def identifier(self, children: typing.Sequence[SassChild]) -> NvidiaSassLabel:
    """Transform a branch/symbol identifier operand into ``NvidiaSassLabel``.

    Args:
        children: Identifier token.

    Returns:
        NvidiaSassLabel: Constructed label operand CST node.
    """
    return NvidiaSassLabel(name=_tok_val(children[0]), leading_trivia=_get_trivia(children[0]))


class NvidiaSassParser(IsaParser[NvidiaSassModule, NvidiaSassNode]):
  """Facade for parsing NVIDIA SASS source text into ``NvidiaSassModule`` CSTs.

  Attributes:
      SPEC: NVIDIA SASS lexical and grammar dialect specification.
      LEXER_CLS: Lexer class bound to Lark Earley construction.
  """

  SPEC: typing.ClassVar[IsaDialectSpec] = NVIDIA_SASS_SPEC
  LEXER_CLS: typing.ClassVar[type[NvidiaSassLexer]] = NvidiaSassLexer

  def _build_transformer(self) -> Transformer[Token, NvidiaSassNode]:
    """Instantiate ``NvidiaSassTransformer``.

    Returns:
        Transformer[Token, NvidiaSassNode]: Dialect transformer instance.
    """
    return NvidiaSassTransformer()

  def _empty_module(self, statements: list[NvidiaSassNode] | None = None) -> NvidiaSassModule:
    """Construct a ``NvidiaSassModule``.

    Args:
        statements: Optional parsed statements.

    Returns:
        NvidiaSassModule: Constructed module node.
    """
    return NvidiaSassModule(statements=statements)

  def _validate_module(self, node: NvidiaSassNode | Tree[Token]) -> tuple[NvidiaSassModule, list[NvidiaSassNode]]:
    """Validate that Lark reduction produced a ``NvidiaSassModule``.

    Args:
        node: Root object produced by ``NvidiaSassTransformer``.

    Returns:
        tuple[NvidiaSassModule, list[NvidiaSassNode]]: Validated module and its statement list.

    Raises:
        SyntaxParseError: If ``node`` is not a ``NvidiaSassModule``.
    """
    if not isinstance(node, NvidiaSassModule):
      raise SyntaxParseError(f"Expected NvidiaSassModule root, got {type(node).__name__}")
    return node, list(node.statements)


__all__ = [
  "NvidiaSassComment",
  "NvidiaSassDirective",
  "NvidiaSassImmediate",
  "NvidiaSassInstruction",
  "NvidiaSassLabel",
  "NvidiaSassLexer",
  "NvidiaSassMemory",
  "NvidiaSassModule",
  "NvidiaSassOperand",
  "NvidiaSassParser",
  "NvidiaSassPredicate",
  "NvidiaSassRegister",
  "NvidiaSassToken",
  "NvidiaSassTransformer",
  "Trivia",
  "_get_trivia",
]
