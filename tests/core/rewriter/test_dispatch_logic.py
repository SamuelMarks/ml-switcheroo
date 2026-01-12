"""
Tests for Conditional API Dispatch in PivotRewriter.

Verifies that:
1.  Runtime rules trigger API switching based on argument values.
2.  Argument values are extracted from literals correctly.
3.  Positional vs Keyword mapping is respected (Std Name resolution).
4.  Condition logic (EQ, GT, IN) works.
5.  **NEW**: Type Matching (IS_TYPE) works for lists vs scalars.
"""

import pytest
import libcst as cst

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.dsl import Rule, LogicOp


class MockDispatchSemantics(SemanticsManager):
  def __init__(self):
    # Skip super
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}

    # Define 'resize' with dispatch rules
    resize_def = {
      "std_args": ["image", "dummy", "mode"],
      "variants": {
        "torch": {"api": "torch.resize", "args": {}},
        "jax": {
          "api": "jax.image.resize",  # Default
          "args": {},
          "dispatch_rules": [
            Rule(
              if_arg="mode",
              op=LogicOp.EQ,
              val="nearest",
              use_api="jax.image.resize_nearest",
            ),
            Rule(
              if_arg="mode",
              op=LogicOp.IN,
              val=["bilinear", "bicubic"],
              use_api="jax.image.resize_bi",
            ),
          ],
        },
      },
    }
    self.data["resize"] = resize_def
    self._reverse_index["torch.resize"] = ("resize", resize_def)

    # Define generic op for numeric check
    clamp_def = {
      "std_args": ["x", "limit"],
      "variants": {
        "torch": {"api": "torch.clamp"},
        "jax": {
          "api": "jnp.clip",
          "dispatch_rules": [
            Rule(
              if_arg="limit",
              op=LogicOp.GT,
              val=100,
              use_api="jnp.heavy_clip",
            )
          ],
        },
      },
    }
    self.data["clamp"] = clamp_def
    self._reverse_index["torch.clamp"] = ("clamp", clamp_def)

    # Define op with Type-Based Dispatch
    process_def = {
      "std_args": ["data"],
      "variants": {
        "torch": {"api": "torch.process"},
        "jax": {
          "api": "jax.single_process",
          "dispatch_rules": [
            Rule(
              if_arg="data",
              op=LogicOp.IS_TYPE,
              val="list",
              use_api="jax.batch_process",
            ),
            Rule(
              if_arg="data",
              op=LogicOp.IS_TYPE,
              val="int",
              use_api="jax.int_process",
            ),
          ],
        },
      },
    }
    self.data["process"] = process_def
    self._reverse_index["torch.process"] = ("process", process_def)

  def get_definition(self, name):
    # Heuristic lookup to handle method calls like "x.clamp"
    if name.endswith("resize"):
      return ("resize", self.data["resize"])
    if name.endswith("clamp"):
      return ("clamp", self.data["clamp"])
    if name.endswith("process"):
      return ("process", self.data["process"])
    return self._reverse_index.get(name)

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter():
  semantics = MockDispatchSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics, config)


def rewrite(rewriter, code):
  tree = cst.parse_module(code)
  # Fix: Use pipeline conversion
  return rewriter.convert(tree).code


def test_dispatch_equality_string(rewriter):
  """
  Scenario: resize(..., mode='nearest')
  Expect: jax.image.resize_nearest
  """
  code = "y = torch.resize(x, None, mode='nearest')"
  res = rewrite(rewriter, code)

  assert "jax.image.resize_nearest" in res
  assert "mode='nearest'" in res  # Args preserved, just API swapped


def test_dispatch_fallback_default(rewriter):
  """
  Scenario: resize(..., mode='linear') -> No rule match.
  Expect: jax.image.resize (default api)
  """
  code = "y = torch.resize(x, None, mode='linear')"
  res = rewrite(rewriter, code)

  assert "jax.image.resize(" in res


def test_dispatch_in_list(rewriter):
  """
  Scenario: resize(..., mode='bicubic') -> Matches IN list.
  Expect: jax.image.resize_bi
  """
  code = "y = torch.resize(x, None, mode='bicubic')"
  res = rewrite(rewriter, code)

  assert "jax.image.resize_bi" in res


def test_dispatch_positional_extraction(rewriter):
  """
  Scenario: torch.resize(x, None, 'nearest')
  'mode' is 3rd argument in std_args.
  Value 'nearest' is positional arg 2.
  Expect: Dispatch triggers.
  """
  code = "y = torch.resize(x, None, 'nearest')"
  res = rewrite(rewriter, code)

  assert "jax.image.resize_nearest" in res


def test_dispatch_numeric_gt(rewriter):
  """
  Scenario: torch.clamp(x, 150) -> limit=150 > 100.
  Expect: jnp.heavy_clip
  """
  code = "y = torch.clamp(x, 150)"
  res = rewrite(rewriter, code)

  assert "jnp.heavy_clip" in res


def test_dispatch_numeric_method_call(rewriter):
  """
  Scenario: x.clamp(150). Method call implicit self.
  std_args: [x, limit]. Method args: [limit].
  Positional index logic should map limit -> arg 0.
  Expect: jnp.heavy_clip
  """
  code = "y = x.clamp(50)"  # < 100
  res = rewrite(rewriter, code)
  assert "jnp.clip" in res  # Default

  code2 = "y = x.clamp(150)"  # > 100
  res2 = rewrite(rewriter, code2)
  assert "jnp.heavy_clip" in res2


def test_dispatch_is_type_list(rewriter):
  """
  Scenario: input is List Literal [1, 2] -> Is List -> Batch API.
  """
  code = "y = torch.process([1, 2])"
  res = rewrite(rewriter, code)
  assert "jax.batch_process" in res


def test_dispatch_is_type_int(rewriter):
  """
  Scenario: input is Integer Literal 5 -> Is Int -> Int API.
  """
  code = "y = torch.process(5)"
  res = rewrite(rewriter, code)
  assert "jax.int_process" in res


def test_dispatch_is_type_fallback(rewriter):
  """
  Scenario: input is variable 'x' (unknown type/Name).
  Logic fails integer/list checks -> Default API.
  """
  code = "y = torch.process(x)"
  res = rewrite(rewriter, code)
  assert "jax.single_process" in res
