"""Test suite for the Arg Packing module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return typing.cast(str, new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")
    return ""


def get_rewriter_for_target(target_fw: str, pack_kw: str, pack_as: typing.Optional[str] = None) -> PivotRewriter:
  """Gets rewriter for target."""
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  variant: dict[str, typing.Any] = {"api": "target.transpose", "pack_to_tuple": pack_kw}
  if pack_as:
    variant["pack_as"] = pack_as
  permute_def: dict[str, typing.Any] = {
    "std_args": ["x", {"name": "axes", "is_variadic": True}],
    "variants": {"torch": {"api": "torch.permute"}, target_fw: variant},
  }

  def get_def_side_effect(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def side effect."""
    if name == "torch.permute":
      return ("permute_dims", permute_def)
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_definition_by_id.return_value = permute_def
  mgr.get_known_apis.return_value = {"permute_dims": permute_def}
  mgr.is_verified.return_value = True

  def resolve_variant(abstract_id: str, framework: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves variant."""
    if abstract_id == "permute_dims" and framework == target_fw:
      return typing.cast(dict[str, typing.Any], permute_def["variants"][target_fw])
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.get_framework_config.return_value = {}
  cfg = RuntimeConfig(source_framework="torch", target_framework=target_fw)
  return PivotRewriter(semantics=mgr, config=cfg)


def test_generic_axis_packing_tuple() -> None:
  """Verifies the behavior of generic axis packing tuple."""
  rewriter = get_rewriter_for_target("jax", pack_kw="axes")
  code: str = "y = torch.permute(x, 2, 0, 1)"
  result: str = rewrite_code(rewriter, code)
  assert "target.transpose" in result
  clean: str = result.replace(" ", "")
  assert "axes=(2,0,1)" in clean


def test_custom_perm_packing_list() -> None:
  """Verifies the behavior of custom perm packing list."""
  rewriter = get_rewriter_for_target("tensorflow", pack_kw="perm", pack_as="List")
  code: str = "y = torch.permute(x, 0, 2, 1)"
  result: str = rewrite_code(rewriter, code)
  clean: str = result.replace(" ", "")
  assert "perm=[0,2,1]" in clean


def test_pack_single_dim_list() -> None:
  """Verifies the behavior of pack single dim list."""
  rewriter = get_rewriter_for_target("jax", pack_kw="axes", pack_as="List")
  code: str = "y = torch.permute(x, 0)"
  result: str = rewrite_code(rewriter, code)
  clean: str = result.replace(" ", "")
  assert "axes=[0]" in clean
