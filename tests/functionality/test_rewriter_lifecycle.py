"""
Tests for Model Lifecycle Translation (Framework Specific Idioms), Version Enforcement,
and Deprecation Warnings.

Verifies Feature 06:
1. Stripping of tensor movement methods (.to(), .cpu(), .cuda(), .detach()).
2. Warning/Stubbing of model mode methods (.eval(), .train()).
3. Correct handling of chained calls (e.g., model.eval().to(device)).
4. **Version Constraints**: Warnings generated when target version is incompatible.
5. **Deprecation**: Warnings generated when using deprecated ops.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.base import BaseRewriter # For mocking

class MockSemantics(SemanticsManager):
  """Minimal semantics manager with Trait Support."""

  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self._known_rng_methods = set()
    self._validation_status = {} # Added to prevent attribute error in base

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
      }
    }

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def is_verified(self, _id):
    return True

  def _inject(self, name, s_api, t_api, min_v=None, max_v=None, deprecated=False, replaced_by=None):
    tgt_var = {"api": t_api}
    if min_v: tgt_var["min_version"] = min_v
    if max_v: tgt_var["max_version"] = max_v

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
  return PivotRewriter(semantics, config)

def rewrite(rewriter, code):
  """Executes the rewriter on the code string."""
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
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

def test_strip_cpu_cuda(rewriter):
  """
  Input: y = x.cpu().cuda()
  Effect: Both stripped.
  Output: y = x
  """
  code = "y = x.cpu().cuda()"
  result = rewrite(rewriter, code)

  is_logical_cpu = any(".cpu" in line and not line.strip().startswith("#") for line in result.splitlines())
  assert not is_logical_cpu
  assert "y = x" in result
  assert "Stripped framework-specific lifecycle method" in result

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

def test_chaining_mixed(rewriter):
  """
  Input: z = torch.abs(t).to(d)
  Effect:
      1. torch.abs(t) -> jax.numpy.abs(t) [Standard Rewrite]
      2. .to(d) -> Identity [Lifecycle Strip]
  Output: z = jax.numpy.abs(t)
  """
  code = "z = torch.abs(t).to(d)"
  result = rewrite(rewriter, code)

  # Standard rewrite should happen
  assert "jax.numpy.abs(t)" in result

  # .to() should be gone from code logic
  is_to = any(".to(" in line and not line.strip().startswith("#") for line in result.splitlines())
  assert not is_to
  # Semantics preserved
  assert "z =" in result

def test_unknown_method_passed_through(rewriter):
  """
  Input: x.my_method()
  Effect: Preserved. No warning.
  """
  code = "y = x.my_method()"
  result = rewrite(rewriter, code)

  assert "x.my_method()" in result
  assert EscapeHatch.START_MARKER not in result

def test_argument_cleaning_in_strip(rewriter):
  """
  Input: x.to(device='cuda', dtype=torch.float32)
  Effect: Arguments inside stripped call are removed entirely.
  """
  code = "y = x.to(device='cuda', dtype=torch.float32)"
  result = rewrite(rewriter, code)

  # Output logic should be 'y = x'
  assert "y = x" in result

  # Check that arguments are gone from generated code
  # We added 'float32' to mock semantics so it shouldn't trigger an error rollback.
  is_cuda = any("'cuda'" in line and not line.strip().startswith("#") for line in result.splitlines())
  assert not is_cuda


def test_version_constraint_check_min(rewriter):
    """
    Scenario: Op requires min_version="9.0.0". Target is "1.0.0".
    Expectation: Warning generated.
    """
    # Current version is 1.0.0 in MockSemantics

    code = "y = torch.future(x)"
    result = rewrite(rewriter, code)

    # Transformation happens
    assert "jax.future(x)" in result

    # Warning attached
    assert EscapeHatch.START_MARKER in result
    assert "Target jax@1.0.0 is older than required 9.0.0" in result

def test_version_constraint_check_max(rewriter):
    """
    Scenario: Op requires max_version="0.0.1". Target is "1.0.0".
    Expectation: Warning generated.
    """
    code = "y = torch.legacy(x)"
    result = rewrite(rewriter, code)

    assert "jax.legacy(x)" in result

    assert EscapeHatch.START_MARKER in result
    assert "Target jax@1.0.0 exceeds max supported 0.0.1" in result

def test_version_constraint_pass(rewriter):
    """
    Scenario: No constraints or compatible constraints.
    """
    # Create op with compatible constraints
    rewriter.semantics._inject("compat", "torch.compat", "jax.compat", min_v="0.5.0", max_v="2.0.0")

    code = "y = torch.compat(x)"
    result = rewrite(rewriter, code)

    assert "jax.compat(x)" in result
    assert EscapeHatch.START_MARKER not in result

@patch("importlib.metadata.version")
def test_live_version_lookup(mock_ver, rewriter):
    """
    Verify version check logic falls back to importlib if config missing.
    """
    # Remove config version to force live lookup
    rewriter.semantics.framework_configs["jax"] = {}

    mock_ver.return_value = "0.0.1"

    # Test Min Failure (Require 9.0)
    code = "y = torch.future(x)"
    result = rewrite(rewriter, code)
    assert "Target jax@0.0.1 is older than required 9.0.0" in result
    mock_ver.assert_called_with("jax")

def test_deprecation_warning(rewriter):
    """
    Scenario: Op marked as deprecated.
    Expectation: Warning generated.
    """
    code = "y = torch.unsafe(x)"
    result = rewrite(rewriter, code)

    assert "jax.unsafe(x)" in result
    assert "Usage of deprecated operation 'unsafe_op'" in result

def test_deprecation_replacement_suggestion(rewriter):
    """
    Scenario: Op marked as deprecated with replaced_by.
    Expectation: Warning mentions replacement.
    """
    code = "y = torch.old_scatter(x)"
    result = rewrite(rewriter, code)

    assert "jax.scatter(x)" in result
    assert "Consider using 'Scatter' instead" in result
