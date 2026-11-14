"""Type stubs for docutils.nodes."""

class Node:
  """Base Docutils document tree node."""

class raw(Node):
  """Raw pass-through Docutils node."""

  def __init__(self, rawsource: str = ..., text: str = ..., *, format: str = ...) -> None:
    """Initialize raw node."""
