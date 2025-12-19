import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  squeeze_def = {
    "std_args": ["input", "dim"],
    "variants": {"torch": {"api": "torch.squeeze"}, "jax": {"api": "jax.numpy.squeeze", "args": {"dim": "axis"}}},
  }
  unsqueeze_def = {
    "std_args": ["input", "dim"],
    "variants": {"torch": {"api": "torch.unsqueeze"}, "jax": {"api": "jax.numpy.expand_dims", "args": {"dim": "axis"}}},
  }

  def get_def(name):
    if "unsqueeze" in name:
      return ("Unsqueeze", unsqueeze_def)
    if "squeeze" in name:
      return ("Squeeze", squeeze_def)
    return None

  def resolve(aid, fw):
    if aid == "Unsqueeze" and fw == "jax":
      return unsqueeze_def["variants"]["jax"]
    if aid == "Squeeze" and fw == "jax":
      return squeeze_def["variants"]["jax"]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Squeeze": squeeze_def, "Unsqueeze": unsqueeze_def}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_unsqueeze_mapping(rewriter):
  code = "y = torch.unsqueeze(x, dim=1)"
  res = rewrite_code(rewriter, code)
  assert "jax.numpy.expand_dims" in res
  assert "axis=1" in res
  # Assert 'dim=' keyword is not present
  assert "dim=" not in res


def test_squeeze_mapping(rewriter):
  code = "y = torch.squeeze(x, dim=2)"
  res = rewrite_code(rewriter, code)
  assert "jax.numpy.squeeze" in res
  assert "axis=2" in res


def test_method_to_function_unsqueeze(rewriter):
  code = "y = x.unsqueeze(0)"
  res = rewrite_code(rewriter, code)
  assert "jax.numpy.expand_dims" in res
  assert "(x, 0)" in res
