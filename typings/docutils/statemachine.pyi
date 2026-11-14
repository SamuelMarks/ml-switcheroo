"""Type stubs for docutils.statemachine."""

from typing import Sequence

class StringList:
  """Docutils line-numbered string sequence."""

  def __init__(
    self,
    initlist: Sequence[str] | None = ...,
    source: str | None = ...,
  ) -> None:
    """Initialize StringList."""
