"""
Tests for Model Lifecycle Translation (Framework Specific Idioms).

Verifies Feature 06:
1. Stripping of tensor movement methods (.to(), .cpu(), .cuda(), .detach()).
2. Warning/Stubbing of model mode methods (.eval(), .train()).
3. Correct handling of chained calls (e.g., model.eval().to(device)).
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """Minimal semantics manager for rewriter tests."""

  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}

    # Add a basic op to ensure standard rewrites still work alongside stripping
    self._inject("abs", "torch.abs", "jax.numpy.abs")
    # Add basic types to prevent Attribute lookup failures in strict mode
    self._inject("float32", "torch.float32", "jax.numpy.float32")

  def _inject(self, name, s_api, t_api):
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
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
