import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def get_rewriter_for_target(target_fw, pack_kw, pack_as=None):
  """
  Creates a rewriter with mocked semantics.
  If pack_as is set ('List' or 'Tuple'), it is injected into the variant.
  Note: This test uses the *Core Normalization* packing logic by NOT requesting the plugin,
  thus verifying the generalizability of the feature implemented in `normalization.py`.
  """
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  variant = {"api": "target.transpose", "pack_to_tuple": pack_kw}
  if pack_as:
    variant["pack_as"] = pack_as

  # Definition that uses core packing (no Requires Plugin)
  permute_def = {
    "std_args": ["x", {"name": "axes", "is_variadic": True}],
    "variants": {
      "torch": {"api": "torch.permute"},
      target_fw: variant,
    },
  }

  # Mock lookup
  def get_def_side_effect(name):
    if name == "torch.permute":
      return "permute_dims", permute_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_definition_by_id.return_value = permute_def
  mgr.get_known_apis.return_value = {"permute_dims": permute_def}
  mgr.is_verified.return_value = True

  # Mock resolution
  def resolve_variant(abstract_id, framework):
    if abstract_id == "permute_dims" and framework == target_fw:
      return permute_def["variants"][target_fw]
    return None

  mgr.resolve_variant.side_effect = resolve_variant

  # Fix: Ensure get_framework_config returns safe defaults
  mgr.get_framework_config.return_value = {}

  cfg = RuntimeConfig(source_framework="torch", target_framework=target_fw)
  return PivotRewriter(semantics=mgr, config=cfg)


def test_generic_axis_packing_tuple():
  """Verify default packing to 'axes' (JAX style default) using Tuple."""
  rewriter = get_rewriter_for_target("jax", pack_kw="axes")

  code = "y = torch.permute(x, 2, 0, 1)"
  result = rewrite_code(rewriter, code)

  assert "target.transpose" in result
  # "y=target.transpose(x,axes=(2,0,1))"
  clean = result.replace(" ", "")
  assert "axes=(2,0,1)" in clean


def test_custom_perm_packing_list():
  """
  Verify packing to 'perm' into a List (e.g. for list-based APIs like torch.cat).
  Features: pack_as="List".
  """
  rewriter = get_rewriter_for_target("tensorflow", pack_kw="perm", pack_as="List")

  code = "y = torch.permute(x, 0, 2, 1)"
  result = rewrite_code(rewriter, code)

  clean = result.replace(" ", "")
  # Must use square brackets
  assert "perm=[0,2,1]" in clean


def test_pack_single_dim_list():
  """Verify single arg packing into a List container."""
  rewriter = get_rewriter_for_target("jax", pack_kw="axes", pack_as="List")

  code = "y = torch.permute(x, 0)"
  result = rewrite_code(rewriter, code)
  clean = result.replace(" ", "")
  # List packing should be [0] not [0,]
  assert "axes=[0]" in clean
