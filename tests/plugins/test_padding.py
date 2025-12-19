"""
Tests for Padding Normalization Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.padding import transform_padding


def rewrite_code(rewriter, code):
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter():
  # Register the hook manually to bypass discovery
  hooks._HOOKS["padding_converter"] = transform_padding
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define 'Pad' operation abstract
  pad_def = {
    "variants": {
      "torch": {"api": "torch.nn.functional.pad"},
      "jax": {"api": "jnp.pad", "requires_plugin": "padding_converter"},
    }
  }

  # Mock Manager Lookups
  def get_def(name):
    if "pad" in name:
      return ("Pad", pad_def)
    return None

  def resolve(aid, fw):
    if aid == "Pad" and fw == "jax":
      return pad_def["variants"]["jax"]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  # Context lookup uses get_known_apis
  mgr.get_known_apis.return_value = {"Pad": pad_def}
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_padding_2d_nchw(rewriter):
  """
  Input: F.pad(x, (1, 2, 3, 4))  -> Left=1, Right=2, Top=3, Bottom=4
  Output: jnp.pad(x, ((0, 0), (0, 0), (3, 4), (1, 2)))
  """
  code = "y = F.pad(x, (1, 2, 3, 4))"
  res = rewrite_code(rewriter, code)

  assert "jnp.pad" in res

  # Flatten structure for easy checking
  clean = res.replace(" ", "").replace("\n", "")
  expected_structure = "((0,0),(0,0),(3,4),(1,2))"

  assert expected_structure in clean


def test_padding_preserves_literals(rewriter):
  """
  Verify variables inside the tuple are preserved.
  Input: F.pad(x, (p1, p2, 3, 4))
  """
  code = "y = F.pad(x, (p1, p2, 3, 4))"
  res = rewrite_code(rewriter, code)

  clean = res.replace(" ", "")
  # Left(p1)/Right(p2) should move to the last tuple
  assert "(p1,p2)" in clean or "(p1,p2))" in clean
  # Top(3)/Bottom(4) should be in the second to last
  assert "(3,4)" in clean


def test_ignore_variable_padding(rewriter):
  """
  Input: F.pad(x, padding_var)
  Expectation: No change (cannot statically decompose).
  """
  code = "y = F.pad(x, padding_var)"
  res = rewrite_code(rewriter, code)

  # Name might swap if base rewriter handles it,
  # but structure should remain flat argument.
  # The plugin returns original node if check fails,
  # letting BaseRewriter rename IF configured, but here plugin handles rename.
  # If plugin returns original node, no rename happens inside plugin.

  # Verify that the plugin aborted deep rewrite
  assert "((0, 0)" not in res
