"""Test suite for the Dispatch Logic module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.dsl import Rule, LogicOp


class MockDispatchSemantics(SemanticsManager):
  """Mock Dispatch Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockDispatchSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}
    resize_def = {
      "std_args": ["image", "dummy", "mode"],
      "variants": {
        "torch": {"api": "torch.resize", "args": {}},
        "jax": {
          "api": "jax.image.resize",
          "args": {},
          "dispatch_rules": [
            Rule(if_arg="mode", op=LogicOp.EQ, val="nearest", use_api="jax.image.resize_nearest"),
            Rule(if_arg="mode", op=LogicOp.IN, val=["bilinear", "bicubic"], use_api="jax.image.resize_bi"),
          ],
        },
      },
    }
    self.data["resize"] = resize_def
    self._reverse_index["torch.resize"] = ("resize", resize_def)
    clamp_def = {
      "std_args": ["x", "limit"],
      "variants": {
        "torch": {"api": "torch.clamp"},
        "jax": {
          "api": "jnp.clip",
          "dispatch_rules": [Rule(if_arg="limit", op=LogicOp.GT, val=100, use_api="jnp.heavy_clip")],
        },
      },
    }
    self.data["clamp"] = clamp_def
    self._reverse_index["torch.clamp"] = ("clamp", clamp_def)
    process_def = {
      "std_args": ["data"],
      "variants": {
        "torch": {"api": "torch.process"},
        "jax": {
          "api": "jax.single_process",
          "dispatch_rules": [
            Rule(if_arg="data", op=LogicOp.IS_TYPE, val="list", use_api="jax.batch_process"),
            Rule(if_arg="data", op=LogicOp.IS_TYPE, val="int", use_api="jax.int_process"),
          ],
        },
      },
    }
    self.data["process"] = process_def
    self._reverse_index["torch.process"] = ("process", process_def)

  def get_definition(self, name):
    """Mock implementation of get definition."""
    if name.endswith("resize"):
      return ("resize", self.data["resize"])
    if name.endswith("clamp"):
      return ("clamp", self.data["clamp"])
    if name.endswith("process"):
      return ("process", self.data["process"])
    return self._reverse_index.get(name)

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  semantics = MockDispatchSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite(rewriter, code):
  """Rewrites ."""
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_dispatch_equality_string(rewriter):
  """Verifies the behavior of dispatch equality string."""
  code = "y = torch.resize(x, None, mode='nearest')"
  res = rewrite(rewriter, code)
  assert "jax.image.resize_nearest" in res
  assert "mode='nearest'" in res


def test_dispatch_fallback_default(rewriter):
  """Verifies the behavior of dispatch fallback default."""
  code = "y = torch.resize(x, None, mode='linear')"
  res = rewrite(rewriter, code)
  assert "jax.image.resize(" in res


def test_dispatch_in_list(rewriter):
  """Verifies the behavior of dispatch in list."""
  code = "y = torch.resize(x, None, mode='bicubic')"
  res = rewrite(rewriter, code)
  assert "jax.image.resize_bi" in res


def test_dispatch_positional_extraction(rewriter):
  """Verifies the behavior of dispatch positional extraction."""
  code = "y = torch.resize(x, None, 'nearest')"
  res = rewrite(rewriter, code)
  assert "jax.image.resize_nearest" in res


def test_dispatch_numeric_gt(rewriter):
  """Verifies the behavior of dispatch numeric gt."""
  code = "y = torch.clamp(x, 150)"
  res = rewrite(rewriter, code)
  assert "jnp.heavy_clip" in res


def test_dispatch_numeric_method_call(rewriter):
  """Verifies the behavior of dispatch numeric method call."""
  code = "y = x.clamp(50)"
  res = rewrite(rewriter, code)
  assert "jnp.clip" in res
  code2 = "y = x.clamp(150)"
  res2 = rewrite(rewriter, code2)
  assert "jnp.heavy_clip" in res2


def test_dispatch_is_type_list(rewriter):
  """Verifies the behavior of dispatch is type list."""
  code = "y = torch.process([1, 2])"
  res = rewrite(rewriter, code)
  assert "jax.batch_process" in res


def test_dispatch_is_type_int(rewriter):
  """Verifies the behavior of dispatch is type integer."""
  code = "y = torch.process(5)"
  res = rewrite(rewriter, code)
  assert "jax.int_process" in res


def test_dispatch_is_type_fallback(rewriter):
  """Verifies the behavior of dispatch is type fallback."""
  code = "y = torch.process(x)"
  res = rewrite(rewriter, code)
  assert "jax.single_process" in res
