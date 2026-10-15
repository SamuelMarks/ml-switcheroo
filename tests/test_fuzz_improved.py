"""Test suite for the Fuzz Improved module."""

import libcst as cst
from hypothesis import given
from hypothesis import strategies as st

# Base structural syntax to ensure parsing doesn't crash on simple edge cases
SYNTAX_PATTERNS = [
  "pass",
  "x = 1",
  "def f(): pass",
  "class A: pass",
  "if True: pass",
  "while False: pass",
]


@given(code=st.sampled_from(SYNTAX_PATTERNS))
def test_improved_fuzz_parsing(code: str) -> None:
  """Verifies that basic structural patterns parse without error and form valid CSTs."""
  tree = cst.parse_module(code)
  assert tree is not None
  assert isinstance(tree, cst.Module)


def test_cst_transformer_integrity() -> None:
  """Verifies that a basic transformer correctly updates nodes without corrupting the tree."""
  code = "x = y + 1"
  tree = cst.parse_module(code)

  class NameChanger(cst.CSTTransformer):
    """Execute NameChanger test helper."""

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
      """Execute leave_Name test helper."""
      if original_node.value == "y":
        return updated_node.with_changes(value="z")
      return updated_node

  transformed = tree.visit(NameChanger())
  assert transformed.code == "x = z + 1"
  # Ensure original tree is untouched
  assert tree.code == code
