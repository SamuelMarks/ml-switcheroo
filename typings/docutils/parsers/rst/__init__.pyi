"""Type stubs for docutils.parsers.rst."""

from typing import Mapping, Protocol, Sequence

from docutils import nodes
from docutils.statemachine import StringList

class RSTStateProtocol(Protocol):
  """Structural protocol for Docutils RST parser state objects."""

class RSTStateMachineProtocol(Protocol):
  """Structural protocol for Docutils RST parser state-machine objects."""

class Directive:
  """Base class for Docutils reStructuredText directives."""

  has_content: bool

  def __init__(
    self,
    name: str,
    arguments: Sequence[str],
    options: Mapping[str, str | int | bool | None],
    content: StringList,
    lineno: int,
    content_offset: int,
    block_text: str,
    state: RSTStateProtocol,
    state_machine: RSTStateMachineProtocol,
  ) -> None:
    """Initialize Directive."""

  def run(self) -> Sequence[nodes.Node]:
    """Execute directive and return Docutils nodes."""
