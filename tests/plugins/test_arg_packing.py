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


@pytest.fixture
def rewriter():
  # Manually register the hook to ensure it's available for this test context,
  # even though the main logic now uses DSL, the plugin file itself might still
  # contain the hook registration decorator that tests rely on.
  # If the plugin file was removed or emptied, this would fail, but we assume
  # it persists for backward compatibility or is being tested in isolation.
  if hasattr(hooks, "_HOOKS"):
    hooks._HOOKS["pack_varargs"] = pack_varargs
    hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Updated definition using DSL feature instead of plugin for JAX
  # But keep plugin requirement for legacy test if needed?
  # Actually, the failing tests are integration tests using DSL logic.
  # This unit test specifically targeted the plugin function.
  # Let's adjust this test fixture to match the NEW DSL-driven reality
  # where the rewriter handles packing natively if 'pack_to_tuple' is set.

  permute_def = {
    # Must declare is_variadic on an argument for new logic to pick it up
    "std_args": ["x", {"name": "axes", "is_variadic": True}],
    "variants": {
      "torch": {"api": "torch.permute"},
      "jax": {"api": "jax.numpy.transpose", "pack_to_tuple": "axes"},
    },
  }

  def get_def_side_effect(name):
    if name == "torch.permute":
      return "permute_dims", permute_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = {"permute_dims": permute_def}
  mgr.is_verified.return_value = True

  def resolve_variant(abstract_id, target_fw):
    if abstract_id == "permute_dims" and target_fw == "jax":
      return permute_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve_variant

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_pack_permute_vars(rewriter):
  code = "y = torch.permute(x, 2, 0, 1)"
  result = rewrite_code(rewriter, code)
  assert "jax.numpy.transpose" in result

  # Assert string exactness carefully with whitespace
  clean = result.replace(" ", "")
  # "y=jax.numpy.transpose(x,axes=(2,0,1))"
  assert "axes=(2,0,1)" in clean


def test_pack_single_dim(rewriter):
  code = "y = torch.permute(x, 0)"
  result = rewrite_code(rewriter, code)
  clean = result.replace(" ", "")
  assert "axes=(0,)" in clean


def test_preserve_input(rewriter):
  code = "y = torch.permute(my_tensor, 1, 0)"
  result = rewrite_code(rewriter, code)
  assert "jax.numpy.transpose(my_tensor" in result


def test_ignore_wrong_fw(rewriter):
  rewriter.ctx._runtime_config.target_framework = "numpy"
  rewriter.ctx.target_fw = "numpy"
  # Ensure ignore via resolver
  rewriter.semantics.resolve_variant.side_effect = lambda aid, fw: None

  code = "y = torch.permute(x, 1, 0)"
  result = rewrite_code(rewriter, code)
  assert "torch.permute" in result
