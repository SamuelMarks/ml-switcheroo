"""Test suite for the Rewriter Lifecycle module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self._known_rng_methods = set()
    self._validation_status = {}
    self._inject("abs", "torch.abs", "jax.numpy.abs")
    self._inject("float32", "torch.float32", "jax.numpy.float32")
    self._inject("future_op", "torch.future", "jax.future", min_v="9.0.0")
    self._inject("legacy_op", "torch.legacy", "jax.legacy", max_v="0.0.1")
    self._inject("old_scatter", "torch.old_scatter", "jax.scatter", deprecated=True, replaced_by="Scatter")
    self._inject("unsafe_op", "torch.unsafe", "jax.unsafe", deprecated=True)
    self.framework_configs = {
      "torch": {
        "traits": {
          "lifecycle_strip_methods": ["to", "cpu", "cuda", "detach"],
          "lifecycle_warn_methods": ["eval", "train"],
        }
      },
      "jax": {"version": "1.0.0"},
    }

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def is_verified(self, _id):
    """Mock implementation of is verified."""
    return True

  def _inject(self, name, s_api, t_api, min_v=None, max_v=None, deprecated=False, replaced_by=None):
    """Mock implementation of  inject."""
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
  """Provides a mock rewriter for testing."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return TestRewriter(semantics, config)


def rewrite(rewriter, code):
  """Rewrites ."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewriter crashed: {e}")


def test_strip_to_call(rewriter):
  """Verifies the behavior of strip to call."""
  code = "x = tensor.to(device)"
  result = rewrite(rewriter, code)
  assert "x = tensor" in result
  is_to_present = any((".to(" in line and (not line.strip().startswith("#")) for line in result.splitlines()))
  assert not is_to_present
  assert EscapeHatch.START_MARKER in result
  assert "Stripped framework-specific lifecycle method '.to()'" in result


def test_warn_on_eval_train(rewriter):
  """Verifies the behavior of warn on eval train."""
  code = "model.eval()"
  result = rewrite(rewriter, code)
  is_eval = any(("model.eval" in line and (not line.strip().startswith("#")) for line in result.splitlines()))
  assert not is_eval
  assert EscapeHatch.START_MARKER in result
  assert "Ignored model state method '.eval()'" in result


def test_version_constraint_check_min(rewriter):
  """Verifies the behavior of version constraint check min."""
  code = "y = torch.future(x)"
  result = rewrite(rewriter, code)
  assert "jax.future(x)" in result
  assert EscapeHatch.START_MARKER in result
  assert "Target jax@1.0.0 is older than required 9.0.0" in result


def test_deprecation_warning(rewriter):
  """Verifies the behavior of deprecation warning."""
  code = "y = torch.unsafe(x)"
  result = rewrite(rewriter, code)
  assert "jax.unsafe(x)" in result
  assert "Usage of deprecated operation 'unsafe_op'" in result
