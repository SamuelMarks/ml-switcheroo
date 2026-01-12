import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_checks import transform_cuda_check


def rewrite_code(rewriter, code):
  """Executes pipeline."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["cuda_is_available"] = transform_cuda_check
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  op_def = {"variants": {"jax": {"api": "jax.devices", "requires_plugin": "cuda_is_available"}}}

  mgr.get_definition.return_value = ("cuda_is", op_def)
  # Wiring: Only JAX has entries
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_is_available_transform(rewriter):
  code = "if torch.cuda.is_available(): pass"
  res = rewrite_code(rewriter, code)
  assert "len(jax.devices('gpu')) > 0" in res


def test_assignment_transform(rewriter):
  code = "x = torch.cuda.is_available()"
  res = rewrite_code(rewriter, code)
  assert "len(jax.devices('gpu')) > 0" in res


def test_ignore_wrong_fw(rewriter):
  """
  Verify that targeting 'numpy' (which has no wiring for this op)
  results in pass-through.
  """
  # PivotRewriter.target_fw is read-only property from config.
  # To change it, we update the config object in the context.
  rewriter.context.config.target_framework = "numpy"
  # Also update hook context explicitly if it was copied
  rewriter.context.hook_context.target_fw = "numpy"

  # Ensure resolve returns None for numpy (Implicit via fixture logic for 'jax' only)
  rewriter.semantics.resolve_variant.side_effect = lambda a, f: None if f == "numpy" else {}

  code = "x = torch.cuda.is_available()"
  # Should preserve original code
  assert "torch.cuda" in rewrite_code(rewriter, code)
