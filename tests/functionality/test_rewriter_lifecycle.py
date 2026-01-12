"""
Tests for Model Lifecycle Translation, Version Enforcement, and Deprecation Warnings.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """Minimal semantics manager with Trait Support."""

  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self._known_rng_methods = set()
    self._validation_status = {}  # Added to prevent attribute error in base

    # Add a basic op to ensure standard rewrites still work alongside stripping
    self._inject("abs", "torch.abs", "jax.numpy.abs")
    # Add basic types to prevent Attribute lookup failures in strict mode
    self._inject("float32", "torch.float32", "jax.numpy.float32")

    # Version constrained ops
    self._inject("future_op", "torch.future", "jax.future", min_v="9.0.0")
    self._inject("legacy_op", "torch.legacy", "jax.legacy", max_v="0.0.1")

    # Deprecated ops
    self._inject("old_scatter", "torch.old_scatter", "jax.scatter", deprecated=True, replaced_by="Scatter")
    self._inject("unsafe_op", "torch.unsafe", "jax.unsafe", deprecated=True)

    # --- FIX: Populate framework configs for SOURCE traits ---
    self.framework_configs = {
      "torch": {
        "traits": {
          "lifecycle_strip_methods": ["to", "cpu", "cuda", "detach"],
          "lifecycle_warn_methods": ["eval", "train"],
        }
      },
      "jax": {
        # Mock config with version
        "version": "1.0.0"
      },
    }

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def is_verified(self, _id):
    return True

  def _inject(self, name, s_api, t_api, min_v=None, max_v=None, deprecated=False, replaced_by=None):
    tgt_var = {"api": t_api}
    if min_v:
      tgt_var["min_version"] = min_v
    if max_v:
      tgt_var["max_version"] = max_v

    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": tgt_var}, "std_args": ["x"]}

    if deprecated:
      self.data[name]["deprecated"] = True
    if replaced_by:
      self.data[name]["replaced_by"] = replaced_by

    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return TestRewriter(semantics, config)


def rewrite(rewriter, code):
  """Executes the rewriter on the code string."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewriter crashed: {e}")


def test_strip_to_call(rewriter):
  """
  Input: x = tensor.to(device)
  Effect: .to() stripped.
  Output: x = tensor
          (Wrapped in warning markers)
  """
  code = "x = tensor.to(device)"
  result = rewrite(rewriter, code)

  # 1. Check Replacement: 'tensor.to(device)' -> 'tensor'
  assert "x = tensor" in result

  # 2. Check that .to is NOT present in the logic of the code
  is_to_present = any(".to(" in line and not line.strip().startswith("#") for line in result.splitlines())
  assert not is_to_present

  # 3. Check Warning Marker
  assert EscapeHatch.START_MARKER in result
  assert "Stripped framework-specific lifecycle method '.to()'" in result


def test_warn_on_eval_train(rewriter):
  """
  Input: model.eval()
  Effect: .eval() stripped (identity), warning attached.
  Output: model
  """
  code = "model.eval()"
  result = rewrite(rewriter, code)

  # This becomes an expression statement "model" (basically no-op)
  is_eval = any("model.eval" in line and not line.strip().startswith("#") for line in result.splitlines())
  assert not is_eval
  assert EscapeHatch.START_MARKER in result
  assert "Ignored model state method '.eval()'" in result


def test_version_constraint_check_min(rewriter):
  """
  Scenario: Op requires min_version="9.0.0". Target is "1.0.0".
  Expectation: Warning generated.
  """
  code = "y = torch.future(x)"
  result = rewrite(rewriter, code)

  assert "jax.future(x)" in result
  assert EscapeHatch.START_MARKER in result
  assert "Target jax@1.0.0 is older than required 9.0.0" in result


def test_deprecation_warning(rewriter):
  """
  Scenario: Op marked as deprecated.
  Expectation: Warning generated.
  """
  code = "y = torch.unsafe(x)"
  result = rewrite(rewriter, code)

  assert "jax.unsafe(x)" in result
  assert "Usage of deprecated operation 'unsafe_op'" in result
