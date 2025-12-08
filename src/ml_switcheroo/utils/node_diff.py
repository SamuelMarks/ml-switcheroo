"""
AST Node Serialization for Visual Diffs.

This utility module provides mechanisms to convert arbitrary LibCST nodes
textual source code representation "in vacuum". This is essential for
capturing states of the AST before and after transformations without requiring
full file serialization.
"""

import libcst as cst
from typing import Union

# A dummy module used as a context to render detached nodes.
_RENDER_CTX = cst.parse_module("")


def capture_node_source(node: cst.CSTNode) -> str:
  """
  Renders a LibCST node into its Python source code string representation.

  This handles both original nodes (which might carry whitespace info)
  and constructed nodes (detached from the original tree).

  Args:
      node: The CST node to serialise.

  Returns:
      str: The Python code string.
  """
  try:
    # LibCST requires a module context to generate code for a node
    return _RENDER_CTX.code_for_node(node)
  except Exception:
    # Fallback for extreme cases (e.g. malformed or partial nodes)
    return f"<Unrepresentable Node: {type(node).__name__}>"


def diff_nodes(original: cst.CSTNode, modified: cst.CSTNode) -> tuple[str, str, bool]:
  """
  Compares two nodes and returns their source strings if they differ.

  Args:
      original: The node before transformation.
      modified: The node after transformation.

  Returns:
      tuple: (source_before, source_after, has_changed)
  """
  src_before = capture_node_source(original)
  src_after = capture_node_source(modified)

  # We clean whitespace to avoid spamming diffs for simple formatting changes
  # unless exact whitespace preservation is required.
  # For semantic visualizations, logic change is what matters.
  is_diff = src_before.strip() != src_after.strip()

  return src_before, src_after, is_diff
