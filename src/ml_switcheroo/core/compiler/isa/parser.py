"""Table-driven Lexer, Trivia Helpers, and Parser Infrastructure for Hardware ISAs.

Unifies the Lark lexer, trivia preservation, directive/label/operand/immediate
CST construction, and Lark parser caching shared across NVIDIA SASS, AMD RDNA,
and future hardware assembly dialects.
"""

import re
import typing
from dataclasses import dataclass

from lark import Lark, Transformer, Tree, v_args
from lark.exceptions import LarkError
from lark.lexer import Lexer, Token

from ml_switcheroo.core.compiler.isa.cst import (
  IsaComment,
  IsaDirective,
  IsaImmediate,
  IsaLabel,
  IsaModule,
  IsaNode,
  IsaOperand,
)
from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.exceptions import SyntaxParseError


@dataclass(frozen=True)
class IsaDialectSpec:
  """Declarative grammar and lexical specification for a hardware ISA dialect.

  Attributes:
      token_regex: Ordered ``(token_kind, regex_pattern)`` rules.
      punctuation_map: Mapping from literal punctuation symbols to Lark terminal names.
      grammar: Lark EBNF grammar specification string.
      emit_eof_trivia: Whether trailing whitespace at EOF emits an ``EOF_TRIVIA`` token.
      line_by_line: Whether ``IsaParser.parse`` parses non-empty lines individually.
  """

  token_regex: tuple[tuple[str, str], ...]
  punctuation_map: typing.Mapping[str, str]
  grammar: str
  emit_eof_trivia: bool = False
  line_by_line: bool = False

  @property
  def compiled_regex(self) -> str:
    """Build the combined named-group regular expression pattern from ``token_regex``.

    Returns:
        str: Combined named-group regular expression string.
    """
    return "|".join(f"(?P<{name}>{pattern})" for name, pattern in self.token_regex)


@typing.runtime_checkable
class LexerTextProtocol(typing.Protocol):
  """Protocol for Lark ``LexerState`` objects exposing ``text``.

  Attributes:
      text: Underlying source buffer string (`str`).
  """

  text: str


class IsaTextInputProtocol(typing.Protocol):
  """Protocol for Lark lexer/parser inputs and parse-tree nodes convertible to ``str``."""

  def __str__(self) -> str:
    """Return string representation.

    Returns:
        str: String representation.
    """
    ...


IsaChild: typing.TypeAlias = (
  Token | Tree[Token] | IsaNode | typing.Sequence[IsaTextInputProtocol] | IsaTextInputProtocol | str | None
)


def tok_val(node: IsaChild) -> str:
  """Extract the string value from a Lark ``Token`` or parse-tree child.

  Args:
      node: Lark token or child object.

  Returns:
      str: Extracted string value.
  """
  return str(node.value) if isinstance(node, Token) else str(node)


class IsaToken(Token):
  """Custom Lark ``Token`` carrying leading ``Trivia`` items.

  Attributes:
      leading_trivia: Trivia elements preceding this token (`list[Trivia]`).
  """

  __slots__ = ("leading_trivia",)

  leading_trivia: list[Trivia]


TokenT = typing.TypeVar("TokenT", bound=IsaToken)


class IsaLexer(Lexer, typing.Generic[TokenT]):
  """Parameterized table-driven Lark ``Lexer`` driven by an ``IsaDialectSpec``.

  Attributes:
      SPEC: Dialect lexical specification (`typing.ClassVar[IsaDialectSpec]`).
      lexer_conf: Lark lexer configuration passed at construction.
  """

  SPEC: typing.ClassVar[IsaDialectSpec]

  def __init__(self, lexer_conf: IsaTextInputProtocol | None) -> None:
    """Initialize the table-driven ISA lexer.

    Args:
        lexer_conf: Configuration object provided by Lark.
    """
    self.lexer_conf: IsaTextInputProtocol | None = lexer_conf

  def _make_token(self, kind: str, value: str, leading: list[Trivia]) -> TokenT:
    """Create a dialect token carrying ``leading`` trivia.

    Args:
        kind: Token terminal type name.
        value: Matched token substring.
        leading: Accumulated leading trivia list.

    Returns:
        TokenT: Constructed dialect token.

    Raises:
        NotImplementedError: Must be implemented by concrete ``IsaLexer`` subclasses.
    """
    raise NotImplementedError

  def lex(
    self,
    lexer_state: IsaTextInputProtocol,
    parser_state: IsaTextInputProtocol | None = None,
  ) -> typing.Iterator[TokenT]:
    """Tokenize ``lexer_state`` according to ``SPEC`` while preserving whitespace trivia.

    Args:
        lexer_state: Raw ISA string or Lark ``LexerState`` to tokenize.
        parser_state: Optional Lark parser state.

    Yields:
        TokenT: Dialect tokens carrying leading ``Trivia``.

    Raises:
        SyntaxParseError: If an unrecognized character (``MISMATCH``) is encountered.
    """
    if isinstance(lexer_state, str):
      data = lexer_state
    elif isinstance(lexer_state, LexerTextProtocol):
      data = str(lexer_state.text)
    else:
      data = str(lexer_state)

    leading: list[Trivia] = []
    spec = self.SPEC
    pattern = spec.compiled_regex
    for mo in re.finditer(pattern, data):
      kind = mo.lastgroup or "MISMATCH"
      val = mo.group()
      if kind == "MISMATCH":
        raise SyntaxParseError(f"Unexpected '{val}'")
      if kind == "WS":
        leading.append(Trivia(val))
      elif kind == "COMMENT":
        tok = self._make_token("COMMENT", val, leading)
        leading.clear()
        yield tok
      else:
        if kind == "PUNCTUATION":
          kind = spec.punctuation_map[val]
        tok2 = self._make_token(kind, val, leading)
        leading.clear()
        yield tok2

    if leading and spec.emit_eof_trivia:
      yield self._make_token("EOF_TRIVIA", "", leading)


@typing.runtime_checkable
class LeadingTriviaProtocol(typing.Protocol):
  """Protocol for tokens and CST nodes exposing mutable ``leading_trivia``.

  Attributes:
      leading_trivia: Leading trivia items (`list[Trivia]`).
  """

  leading_trivia: list[Trivia]


@typing.runtime_checkable
class ChildrenNodeProtocol(typing.Protocol):
  """Protocol for parse-tree nodes exposing ``children``.

  Attributes:
      children: Child parse nodes (`typing.Sequence[IsaChild]`).
  """

  children: typing.Sequence[IsaChild]


def get_trivia(
  node: IsaChild | LeadingTriviaProtocol | Tree[IsaToken] | IsaToken | str | int | None,
) -> list[Trivia]:
  """Consume and return ``leading_trivia`` from ``node`` or its first ``Tree`` descendant.

  Args:
      node: Lark ``Tree``, ``Token``, CST node, or scalar value.

  Returns:
      list[Trivia]: Extracted leading trivia items.
  """
  if isinstance(node, LeadingTriviaProtocol):
    res = list(node.leading_trivia)
    node.leading_trivia = []
    return res
  if isinstance(node, Tree) and node.children:
    return get_trivia(node.children[0])
  return []


NodeT = typing.TypeVar("NodeT", bound=IsaNode)
OpT = typing.TypeVar("OpT", bound=IsaOperand)
ModT = typing.TypeVar("ModT", bound=IsaNode)
CommentT = typing.TypeVar("CommentT", bound=IsaComment)
DirectiveT = typing.TypeVar("DirectiveT", bound=IsaDirective)
LabelT = typing.TypeVar("LabelT", bound=IsaLabel)
ImmT = typing.TypeVar("ImmT", bound=IsaImmediate)


class IsaTransformerBase(
  Transformer[Token, NodeT],
  typing.Generic[NodeT, OpT, ModT, CommentT, DirectiveT, LabelT, ImmT],
):
  """Shared Lark ``Transformer`` rules for hardware ISA Concrete Syntax Trees.

  Attributes:
      COMMENT_PREFIX_LEN: Length of the comment token prefix (``2`` for ``//``, ``1`` for ``;``).
      STRIP_COMMENT_TEXT: Whether ``comment_stmt`` strips surrounding whitespace from comment text.
      HOIST_FIRST_STMT_TRIVIA: Whether ``module`` hoists leading trivia from the first statement onto the module.
  """

  COMMENT_PREFIX_LEN: typing.ClassVar[int] = 1
  STRIP_COMMENT_TEXT: typing.ClassVar[bool] = False
  HOIST_FIRST_STMT_TRIVIA: typing.ClassVar[bool] = False

  def _is_statement_node(self, value: IsaChild) -> typing.TypeGuard[NodeT]:
    """Check whether ``value`` is an instance of the dialect's statement CST node type.

    Args:
        value: Candidate child value from Lark reduction.

    Returns:
        bool: ``True`` if ``value`` is a statement CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect transformers.
    """
    raise NotImplementedError

  def _is_operand_node(self, value: IsaChild) -> typing.TypeGuard[OpT]:
    """Check whether ``value`` is an instance of the dialect's operand CST node type.

    Args:
        value: Candidate child value from Lark reduction.

    Returns:
        bool: ``True`` if ``value`` is an operand CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect transformers.
    """
    raise NotImplementedError

  def _create_module(self, statements: list[NodeT], leading: list[Trivia]) -> ModT:
    """Construct the dialect's root module CST node.

    Args:
        statements: Parsed statement CST nodes.
        leading: Module-level leading trivia.

    Returns:
        ModT: Dialect module CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect transformers.
    """
    raise NotImplementedError

  def _create_comment(self, text: str, leading: list[Trivia]) -> CommentT:
    """Construct a dialect comment CST node.

    Args:
        text: Comment body text.
        leading: Leading trivia items.

    Returns:
        CommentT: Dialect comment CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect transformers.
    """
    raise NotImplementedError

  def _create_directive(self, name: str, params: list[str], leading: list[Trivia]) -> DirectiveT:
    """Construct a dialect directive CST node.

    Args:
        name: Directive name without leading dot.
        params: Parsed parameter strings.
        leading: Leading trivia items.

    Returns:
        DirectiveT: Dialect directive CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect transformers.
    """
    raise NotImplementedError

  def _create_label(self, name: str, leading: list[Trivia]) -> LabelT:
    """Construct a dialect label CST node.

    Args:
        name: Label identifier.
        leading: Leading trivia items.

    Returns:
        LabelT: Dialect label CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect transformers.
    """
    raise NotImplementedError

  def _create_immediate(self, value: int | float, is_hex: bool, leading: list[Trivia]) -> ImmT:
    """Construct a dialect immediate CST node.

    Args:
        value: Parsed numeric immediate value.
        is_hex: Whether the literal was written in hexadecimal.
        leading: Leading trivia items.

    Returns:
        ImmT: Dialect immediate CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect transformers.
    """
    raise NotImplementedError

  @v_args(inline=False)
  def module(self, children: typing.Sequence[IsaChild]) -> ModT:
    """Transform the top-level ``module`` grammar production.

    Args:
        children: Parsed statement nodes and optional ``EOF_TRIVIA`` token.

    Returns:
        ModT: Constructed dialect module CST node.
    """
    leading: list[Trivia] = get_trivia(children[0]) if (self.HOIST_FIRST_STMT_TRIVIA and children) else []
    stmts: list[NodeT] = []
    for child in children:
      if self._is_statement_node(child):
        stmts.append(child)
      elif isinstance(child, Token) and child.type == "EOF_TRIVIA":
        eof_trivia = get_trivia(child)
        if stmts:
          stmts[-1].trailing_trivia.extend(eof_trivia)
        else:
          leading.extend(eof_trivia)
    return self._create_module(stmts, leading)

  @v_args(inline=False)
  def comment_stmt(self, children: typing.Sequence[IsaChild]) -> CommentT:
    """Transform a ``COMMENT`` token into a dialect comment CST node.

    Args:
        children: Single-element sequence containing the comment token.

    Returns:
        CommentT: Constructed comment CST node.
    """
    raw_text = tok_val(children[0])[self.COMMENT_PREFIX_LEN :]
    text = raw_text.strip() if self.STRIP_COMMENT_TEXT else raw_text
    return self._create_comment(text, get_trivia(children[0]))

  @v_args(inline=False)
  def directive(self, children: typing.Sequence[IsaChild]) -> DirectiveT:
    """Transform an assembler directive and its parameter list into a directive CST node.

    Args:
        children: Directive dot token, identifier token, and optional parameters.

    Returns:
        DirectiveT: Constructed directive CST node.
    """
    name = tok_val(children[1])
    params: list[str] = []
    if len(children) > 2 and children[2] is not None:
      c2 = children[2]
      param_list = c2.children[0] if isinstance(c2, Tree) and c2.data == "directive_params" and c2.children else c2
      if isinstance(param_list, list):
        for p in param_list:
          if isinstance(p, Token) and p.type == "COMMA":
            continue
          if isinstance(p, ChildrenNodeProtocol):
            params.append("".join(tok_val(x) for x in p.children))
          elif isinstance(p, list):
            params.append("".join(tok_val(x) for x in p))
          elif isinstance(p, Token):
            params.append(p.value)
          else:
            params.append(str(p))
      else:
        params.append(str(param_list))
    return self._create_directive(name, params, get_trivia(children[0]))

  @v_args(inline=False)
  def param_list(self, children: list[IsaChild]) -> list[IsaChild]:
    """Pass through parsed directive parameter elements.

    Args:
        children: Parsed parameter items.

    Returns:
        list[IsaChild]: Unmodified parameter list.
    """
    return children

  @v_args(inline=False)
  def label(self, children: typing.Sequence[IsaChild]) -> LabelT:
    """Transform a label declaration into a dialect label CST node.

    Args:
        children: Label identifier and colon tokens.

    Returns:
        LabelT: Constructed label CST node.
    """
    return self._create_label(tok_val(children[0]), get_trivia(children[0]))

  @v_args(inline=False)
  def operands(self, children: typing.Sequence[IsaChild]) -> list[OpT]:
    """Collect operand CST nodes while attaching ``COMMA`` separator trivia to following operands.

    Args:
        children: Interleaved operand nodes and ``COMMA`` tokens.

    Returns:
        list[OpT]: Operand nodes with comma trivia preserved.
    """
    result: list[OpT] = []
    pending_trivia: list[Trivia] = []
    for child in children:
      if isinstance(child, Token) and child.type == "COMMA":
        pending_trivia.extend(get_trivia(child))
        pending_trivia.append(Trivia(","))
      elif self._is_operand_node(child):
        if pending_trivia:
          child.leading_trivia = pending_trivia + list(child.leading_trivia)
          pending_trivia = []
        result.append(child)
    return result

  def _parse_immediate(
    self,
    val_str: str,
    *,
    leading: list[Trivia],
    is_hex: bool | None = None,
    sign: int = 1,
  ) -> ImmT:
    """Parse ``val_str`` into a dialect immediate CST node with ``sign`` and ``leading`` trivia.

    Args:
        val_str: Numeric literal token text.
        leading: Leading trivia extracted from the first immediate token.
        is_hex: Optional explicit hexadecimal flag; auto-detected from ``0x`` when ``None``.
        sign: Multiplicative sign (``1`` or ``-1``).

    Returns:
        ImmT: Constructed immediate operand CST node.
    """
    hex_flag = ("0x" in val_str.lower()) if is_hex is None else is_hex
    if hex_flag:
      parsed: int | float = sign * int(val_str, 16)
    else:
      parsed = sign * (float(val_str) if "." in val_str else int(val_str, 10))
    return self._create_immediate(parsed, hex_flag, leading)


_LARK_CACHE: dict[type, Lark] = {}

StmtModuleT = typing.TypeVar("StmtModuleT", bound=IsaModule[IsaNode])


class IsaParser(typing.Generic[ModT, NodeT]):
  """Parameterized Lark parser facade shared across hardware ISA dialects.

  Attributes:
      SPEC: Dialect lexical and grammar specification (`typing.ClassVar[IsaDialectSpec]`).
      LEXER_CLS: Dialect ``Lexer`` subclass (`typing.ClassVar[type[Lexer]]`).
      code: Raw assembly source text (`str`).
      parser: Cached Lark Earley parser instance (`Lark`).
      transformer: Dialect CST transformer instance (`Transformer[Token, NodeT]`).
  """

  SPEC: typing.ClassVar[IsaDialectSpec]
  LEXER_CLS: typing.ClassVar[type[Lexer]]

  def __init__(self, code: str) -> None:
    """Initialize the ISA parser for ``code``.

    Args:
        code: Raw assembly source string to parse.
    """
    self.code: str = code
    cls = type(self)
    cached = _LARK_CACHE.get(cls)
    if cached is None:
      cached = Lark(self.SPEC.grammar, parser="earley", lexer=self.LEXER_CLS)
      _LARK_CACHE[cls] = cached
    self.parser: Lark = cached
    self.transformer: Transformer[Token, NodeT] = self._build_transformer()

  def _build_transformer(self) -> Transformer[Token, NodeT]:
    """Instantiate the dialect's Lark ``Transformer``.

    Returns:
        Transformer[Token, NodeT]: Dialect transformer instance.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect parsers.
    """
    raise NotImplementedError

  def _empty_module(self, statements: list[NodeT] | None = None) -> ModT:
    """Construct an empty or statement-populated dialect module CST node.

    Args:
        statements: Optional list of parsed statement nodes.

    Returns:
        ModT: Dialect module CST node.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect parsers.
    """
    raise NotImplementedError

  def _validate_module(self, node: NodeT | Tree[Token]) -> tuple[ModT, list[NodeT]]:
    """Validate that Lark reduction returned the expected dialect module CST node.

    Args:
        node: Transformed root object returned by ``self.transformer.transform``.

    Returns:
        tuple[ModT, list[NodeT]]: Validated module root and its statement list.

    Raises:
        NotImplementedError: Must be implemented by concrete dialect parsers.
    """
    raise NotImplementedError

  def _parse_chunk(self, chunk: str) -> tuple[ModT, list[NodeT]]:
    """Parse a single source string ``chunk`` into a validated module CST node.

    Args:
        chunk: Source text (full module or single line).

    Returns:
        tuple[ModT, list[NodeT]]: Parsed module CST node and its statements.

    Raises:
        SyntaxParseError: If Lark parsing or CST transformation fails.
    """
    try:
      tree = self.parser.parse(chunk)
      node = self.transformer.transform(tree)
      return self._validate_module(node)
    except SyntaxParseError:
      raise
    except (LarkError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
      raise SyntaxParseError(f"Unexpected token: {exc}") from exc

  def parse(self) -> ModT:
    """Parse ``self.code`` into a dialect module CST root.

    Returns:
        ModT: Parsed module CST root node.

    Raises:
        SyntaxParseError: If ``self.code`` contains invalid assembly syntax.
    """
    if not self.code.strip():
      return self._empty_module()

    if not self.SPEC.line_by_line:
      mod, _ = self._parse_chunk(self.code)
      return mod

    statements: list[NodeT] = []
    for line in self.code.splitlines(keepends=True):
      if not line.strip():
        continue
      _, chunk_stmts = self._parse_chunk(line)
      statements.extend(chunk_stmts)
    return self._empty_module(statements)
