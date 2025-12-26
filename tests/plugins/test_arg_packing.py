import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.arg_packing import pack_varargs


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def get_rewriter_for_target(target_fw, pack_kw):
  # Force reload of hooks to pick up any changes if necessary (though registry is global)
  hooks._HOOKS["pack_varargs"] = pack_varargs
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Definition that requires plugin
  permute_def = {
    "std_args": ["x", {"name": "axes", "is_variadic": True}],
    "variants": {
      "torch": {"api": "torch.permute"},
      target_fw: {"api": "target.transpose", "pack_to_tuple": pack_kw, "requires_plugin": "pack_varargs"},
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


def test_generic_axis_packing():
  """Verify default packing to 'axes' (JAX style)."""
  rewriter = get_rewriter_for_target("jax", pack_kw="axes")

  code = "y = torch.permute(x, 2, 0, 1)"
  result = rewrite_code(rewriter, code)

  assert "target.transpose" in result
  # "y=target.transpose(x,axes=(2,0,1))"
  clean = result.replace(" ", "")
  assert "axes=(2,0,1)" in clean


def test_custom_perm_packing():
  """Verify packing to 'perm' (TensorFlow style) driven by data, not hardcoded if."""
  rewriter = get_rewriter_for_target("tensorflow", pack_kw="perm")

  code = "y = torch.permute(x, 0, 2, 1)"
  result = rewrite_code(rewriter, code)

  clean = result.replace(" ", "")
  assert "perm=(0,2,1)" in clean


def test_pack_single_dim():
  """Verify single arg packing."""
  rewriter = get_rewriter_for_target("jax", pack_kw="axes")

  code = "y = torch.permute(x, 0)"
  result = rewrite_code(rewriter, code)
  clean = result.replace(" ", "")
  assert "axes=(0,)" in clean


def test_ignore_wrong_fw():
  """
  Verify plugin is NOT triggered if target framework has no mapping.
  """
  mgr = MagicMock()
  # Resolve returns None
  mgr.resolve_variant.return_value = None
  mgr.get_definition.return_value = None

  mgr.get_framework_config.return_value = {}

  # Valid definition for Torch, but NO mapping for target 'numpy'
  cfg = RuntimeConfig(source_framework="torch", target_framework="numpy")
  rewriter = PivotRewriter(semantics=mgr, config=cfg)

  code = "y = torch.permute(x, 1, 0)"
  # Should not crash, should return original (or rewriter logic skips it)
  result = rewrite_code(rewriter, code)
  assert "torch.permute" in result
