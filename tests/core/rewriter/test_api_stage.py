"""
Tests for ApiStage Execution.

Verifies that the standalone ApiStage correctly applies rewriting logic
using the shared RewriterContext.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.calls.mixer import ApiStage
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self.data["abs"] = {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}
    self.data["float32"] = {"variants": {"jax": {"api": "jax.numpy.float32"}}}
    self.framework_configs = {}
    self.import_data = {}
    self._reverse_index = {}
    self._key_origins = {"abs": "array"}
    self._validation_status = {}

  def get_definition(self, name):
    if name == "torch.abs":
      return ("abs", self.data["abs"])
    if name == "torch.float32":
      return ("float32", self.data["float32"])
    return None

  def resolve_variant(self, aid, fw):
    if aid in self.data and fw in self.data[aid]["variants"]:
      return self.data[aid]["variants"][fw]
    return None

  def is_verified(self, _id):
    return True


@pytest.fixture
def api_stage():
  sem = MockSemantics()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(sem, cfg)
  return ApiStage(ctx)


def test_api_call_rewrite(api_stage):
  """Verify function calls are rewritten."""
  code = "y = torch.abs(x)"
  tree = cst.parse_module(code)

  new_tree = tree.visit(api_stage)

  assert "jnp.abs(x)" in new_tree.code


def test_api_attribute_rewrite(api_stage):
  """Verify attributes are rewritten."""
  code = "d = torch.float32"
  tree = cst.parse_module(code)

  new_tree = tree.visit(api_stage)

  assert "jax.numpy.float32" in new_tree.code


def test_api_assignment_passthrough(api_stage):
  """Verify assignments are visited (MRO check)."""
  code = "x = 1"
  tree = cst.parse_module(code)

  new_tree = tree.visit(api_stage)
  assert new_tree.code == code
