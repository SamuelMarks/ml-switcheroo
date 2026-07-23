"""Semantic Comment Parser for ML-Switcheroo lifters.

This module provides a Lark-based parser for semantic comments generated during
compilation, allowing lifters to reconstruct high-level graphs from assembly
without relying on regex.
"""

from typing import Any, Optional, List
from dataclasses import dataclass
from lark import Lark, Transformer, v_args, Token


@dataclass
class Trivia:
  """Represents non-semantic formatting tokens."""

  text: str

  def to_text(self) -> str:
    """Return the trivia text verbatim."""
    return self.text


class SemanticMarker:
  """Base class for all semantic markers."""

  leading_trivia: str = ""

  def to_text(self) -> str:
    """Reconstructs the original text."""
    raise NotImplementedError()


@dataclass
class SemanticInput(SemanticMarker):
  """Represents an Input marker."""

  name: str
  kw_input_trivia: Optional[Trivia] = None
  name_trivia: Optional[Trivia] = None
  tail: Optional[Trivia] = None

  def to_text(self) -> str:
    """Reconstructs the original text."""
    parts = [self.leading_trivia, "Input"]
    if self.kw_input_trivia:
      parts.append(self.kw_input_trivia.text)
    parts.append(self.name)
    if self.name_trivia:
      parts.append(self.name_trivia.text)
    parts.append("->")
    if self.tail:
      parts.append(self.tail.text)
    return "".join(parts)


@dataclass
class SemanticBegin(SemanticMarker):
  """Represents a BEGIN block marker."""

  kind: str
  id: str
  kw_begin_trivia: Optional[Trivia] = None
  kind_trivia: Optional[Trivia] = None
  lparen_trivia: Optional[Trivia] = None
  id_trivia: Optional[Trivia] = None
  tail: Optional[Trivia] = None

  def to_text(self) -> str:
    """Reconstructs the original text."""
    parts = [self.leading_trivia, "BEGIN"]
    if self.kw_begin_trivia:
      parts.append(self.kw_begin_trivia.text)
    parts.append(self.kind)
    if self.kind_trivia:
      parts.append(self.kind_trivia.text)
    parts.append("(")
    if self.lparen_trivia:
      parts.append(self.lparen_trivia.text)
    parts.append(self.id)
    if self.id_trivia:
      parts.append(self.id_trivia.text)
    parts.append(")")
    if self.tail:
      parts.append(self.tail.text)
    return "".join(parts)


@dataclass
class SemanticEnd(SemanticMarker):
  """Represents an END block marker."""

  kind: str
  id: str
  kw_end_trivia: Optional[Trivia] = None
  kind_trivia: Optional[Trivia] = None
  lparen_trivia: Optional[Trivia] = None
  id_trivia: Optional[Trivia] = None
  tail: Optional[Trivia] = None

  def to_text(self) -> str:
    """Reconstructs the original text."""
    parts = [self.leading_trivia, "END"]
    if self.kw_end_trivia:
      parts.append(self.kw_end_trivia.text)
    parts.append(self.kind)
    if self.kind_trivia:
      parts.append(self.kind_trivia.text)
    parts.append("(")
    if self.lparen_trivia:
      parts.append(self.lparen_trivia.text)
    parts.append(self.id)
    if self.id_trivia:
      parts.append(self.id_trivia.text)
    parts.append(")")
    if self.tail:
      parts.append(self.tail.text)
    return "".join(parts)


@dataclass
class SemanticUnmapped(SemanticMarker):
  """Represents an Unmapped Op marker."""

  api: str
  id: str
  kw_unmapped_trivia: Optional[Trivia] = None
  kw_op_trivia: Optional[Trivia] = None
  api_trivia: Optional[Trivia] = None
  lparen_trivia: Optional[Trivia] = None
  id_trivia: Optional[Trivia] = None
  tail: Optional[Trivia] = None

  def to_text(self) -> str:
    """Reconstructs the original text."""
    parts = [self.leading_trivia, "Unmapped"]
    if self.kw_unmapped_trivia:
      parts.append(self.kw_unmapped_trivia.text)
    parts.append("Op:")
    if self.kw_op_trivia:
      parts.append(self.kw_op_trivia.text)
    parts.append(self.api)
    if self.api_trivia:
      parts.append(self.api_trivia.text)
    parts.append("(")
    if self.lparen_trivia:
      parts.append(self.lparen_trivia.text)
    parts.append(self.id)
    if self.id_trivia:
      parts.append(self.id_trivia.text)
    parts.append(")")
    if self.tail:
      parts.append(self.tail.text)
    return "".join(parts)


@dataclass
class SemanticReturn(SemanticMarker):
  """Represents a Return marker."""

  tail: Optional[Trivia] = None

  def to_text(self) -> str:
    """Reconstructs the original text."""
    parts = [self.leading_trivia, "Return:"]
    if self.tail:
      parts.append(self.tail.text)
    return "".join(parts)


SEMANTIC_GRAMMAR = r"""
    ?start: marker_start

    marker_start: [TRIVIA] marker

    ?marker: input | begin | end | unmapped | return_marker

    KW_INPUT: "Input"
    KW_BEGIN: "BEGIN"
    KW_END: "END"
    KW_UNMAPPED: "Unmapped"
    KW_OP: "Op:"
    KW_RETURN: "Return:"
    LPAREN: "("
    RPAREN: ")"
    ARROW: "->"

    TRIVIA: /[ \t\n\r]+/
    TAIL: /[\s\S]+/

    input: KW_INPUT [TRIVIA] CNAME [TRIVIA] ARROW [TAIL]
    begin: KW_BEGIN [TRIVIA] CNAME [TRIVIA] LPAREN [TRIVIA] CNAME [TRIVIA] RPAREN [TAIL]
    end: KW_END [TRIVIA] CNAME [TRIVIA] LPAREN [TRIVIA] CNAME [TRIVIA] RPAREN [TAIL]
    unmapped: KW_UNMAPPED [TRIVIA] KW_OP [TRIVIA] OP_NAME [TRIVIA] LPAREN [TRIVIA] CNAME [TRIVIA] RPAREN [TAIL]
    return_marker: KW_RETURN [TAIL]

    OP_NAME: /[a-zA-Z0-9_\.]+/

    %import common.CNAME
"""


def _opt_trivia(token: Optional[Token]) -> Optional[Trivia]:
  """Map a Lark Token into a Trivia object."""
  return Trivia(str(token)) if token is not None else None


class _SemanticTransformer(Transformer[Any, Any]):
  """Transforms parsed AST nodes into semantic marker instances."""

  @v_args(inline=False)
  def input(self, children: List[Optional[Token]]) -> SemanticInput:
    """Transform an input rule into a SemanticInput."""
    return SemanticInput(
      name=str(children[2]),
      kw_input_trivia=_opt_trivia(children[1]),
      name_trivia=_opt_trivia(children[3]),
      tail=_opt_trivia(children[5]),
    )

  @v_args(inline=False)
  def begin(self, children: List[Optional[Token]]) -> SemanticBegin:
    """Transform a begin rule into a SemanticBegin."""
    return SemanticBegin(
      kind=str(children[2]),
      id=str(children[6]),
      kw_begin_trivia=_opt_trivia(children[1]),
      kind_trivia=_opt_trivia(children[3]),
      lparen_trivia=_opt_trivia(children[5]),
      id_trivia=_opt_trivia(children[7]),
      tail=_opt_trivia(children[9]),
    )

  @v_args(inline=False)
  def end(self, children: List[Optional[Token]]) -> SemanticEnd:
    """Transform an end rule into a SemanticEnd."""
    return SemanticEnd(
      kind=str(children[2]),
      id=str(children[6]),
      kw_end_trivia=_opt_trivia(children[1]),
      kind_trivia=_opt_trivia(children[3]),
      lparen_trivia=_opt_trivia(children[5]),
      id_trivia=_opt_trivia(children[7]),
      tail=_opt_trivia(children[9]),
    )

  @v_args(inline=False)
  def unmapped(self, children: List[Optional[Token]]) -> SemanticUnmapped:
    """Transform an unmapped rule into a SemanticUnmapped."""
    return SemanticUnmapped(
      api=str(children[4]),
      id=str(children[8]),
      kw_unmapped_trivia=_opt_trivia(children[1]),
      kw_op_trivia=_opt_trivia(children[3]),
      api_trivia=_opt_trivia(children[5]),
      lparen_trivia=_opt_trivia(children[7]),
      id_trivia=_opt_trivia(children[9]),
      tail=_opt_trivia(children[11]),
    )

  @v_args(inline=False)
  def return_marker(self, children: List[Optional[Token]]) -> SemanticReturn:
    """Transform a return_marker rule into a SemanticReturn."""
    return SemanticReturn(
      tail=_opt_trivia(children[1]),
    )

  @v_args(inline=False)
  def marker_start(self, children: List[Any]) -> SemanticMarker:
    """Transform a marker_start rule into a SemanticMarker with leading trivia."""
    from typing import cast

    m = cast(SemanticMarker, children[1])
    m.leading_trivia = str(children[0]) if children[0] else ""
    return m


class SemanticCommentParser:
  """Parses semantic comments."""

  def __init__(self) -> None:
    """Initialize the SemanticCommentParser."""
    self.parser = Lark(SEMANTIC_GRAMMAR, parser="lalr")

  def parse(self, text: str) -> Optional[SemanticMarker]:
    """Parse a semantic comment string into a strongly typed marker.

    Args:
        text: The comment text to parse.

    Returns:
        A SemanticMarker instance if successfully parsed, None otherwise.
    """
    try:
      tree = self.parser.parse(text)
      transformer = _SemanticTransformer()
      marker = transformer.transform(tree)
      return marker  # type: ignore
    except Exception:
      return None
