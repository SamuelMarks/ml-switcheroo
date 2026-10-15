"""Test suite for the Massive Cst Fuzz module."""

import libcst as cst
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Valid CST snippets for invariant testing
VALID_SNIPPETS = [
  "def foo(x): return x + 1",
  "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()",
  "y = torch.relu(x)",
  "z = jax.numpy.add(a, b)",
  "for i in range(10):\n    print(i)",
]

# Invalid syntax strings
INVALID_SNIPPETS = [
  "def foo(x)",
  "class Model(",
  "1 = 2",
  "} { {",
]


@given(code=st.sampled_from(VALID_SNIPPETS))
def test_valid_cst_idempotency(code: str) -> None:
  """Verifies that valid CST parses and re-emits identically without transformers."""
  tree = cst.parse_module(code)
  assert tree.code == code


@given(code=st.sampled_from(INVALID_SNIPPETS))
def test_invalid_syntax_boundaries(code: str) -> None:
  """Verifies that invalid code correctly throws parser errors rather than failing later."""
  with pytest.raises(cst.ParserSyntaxError):
    cst.parse_module(code)


def test_visitor_integrity() -> None:
  """Verifies structural integrity by ensuring visitors do not arbitrarily mutate valid CST."""
  code = "x = torch.nn.functional.relu(y)"
  tree = cst.parse_module(code)

  # Run a simple visitor
  class TrackerVisitor(cst.CSTVisitor):
    """Execute TrackerVisitor test helper."""

    def __init__(self):
      """Execute __init__ test helper."""
      super().__init__()
      self.visited_nodes = 0

    def on_visit(self, node: cst.CSTNode) -> bool:
      """Execute on_visit test helper."""
      self.visited_nodes += 1
      return True

  visitor = TrackerVisitor()
  tree.visit(visitor)

  # Verify the AST remains fully intact
  assert tree.code == code
  assert visitor.visited_nodes > 5
