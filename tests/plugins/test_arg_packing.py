"""Test suite for the Arg Packing module."""

from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return str(new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def get_rewriter_for_target(target_framework: str, pack_kw: str, pack_as: str | None = None) -> PivotRewriter:
  """Gets rewriter for target.

  Args:
      target_framework: The target framework (str) for get rewriter for target.
      pack_kw: The pack kw (str) for get rewriter for target.
      pack_as: The pack as (str | None) for get rewriter for target.

  Returns:
      Resulting `PivotRewriter` from get rewriter for target.
  """
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  variant = {"api": "target.transpose", "pack_to_tuple": pack_kw}
  if pack_as:
    variant["pack_as"] = pack_as
  permute_def = {
    "std_args": ["x", {"name": "axes", "is_variadic": True}],
    "variants": {"torch": {"api": "torch.permute"}, target_framework: variant},
  }

  def get_def_side_effect(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def side effect.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def side effect.
    """
    if name == "torch.permute":
      _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": (getattr({"v": "permute_dims"}, "__getitem__")("v"), getattr({"v": permute_def}, "__getitem__")("v"))},
        "__getitem__",
      )("v")
      return _ret_v
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_definition_by_id.return_value = permute_def
  mgr.get_known_apis.return_value = {"permute_dims": permute_def}
  mgr.is_verified.return_value = True

  def resolve_variant(abstract_id: str, framework: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        abstract_id: The abstract id (str) for resolve variant.
        framework: The framework (str) for resolve variant.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if abstract_id == "permute_dims" and framework == target_framework:
      res = getattr(
        {
          "v": getattr({"v": getattr({"v": permute_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")[
            target_framework
          ]
        },
        "__getitem__",
      )("v")
      return res if isinstance(res, dict) else {}
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.get_framework_config.return_value = {}
  cfg = RuntimeConfig(source_framework="torch", target_framework=target_framework)
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
