"""
Tests for Value-Based Inference in Semantic Harvester.

Verifies that the harvester can infer argument mappings from literal values
(e.g., matching `1` to `axis: int`) even when variable names are generic.
"""

import pytest

from ml_switcheroo.discovery.harvester import SemanticHarvester
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInferenceSemantics(SemanticsManager):
  """
  Mock Manager providing typed definitions.
  """

  def __init__(self):
    # Override to provide deterministic data
    self.data = {
      "sum": {
        "std_args": [("x", "Array"), ("axis", "int")],
        "variants": {
          "jax": {"api": "jax.numpy.sum", "args": {}},
        },
      },
      "mean": {
        "std_args": [("x", "Array"), ("keepdims", "bool")],
        "variants": {
          "jax": {"api": "jax.numpy.mean"},
        },
      },
      "ambiguous_op": {
        "std_args": [("a", "int"), ("b", "int")],
        "variants": {
          "jax": {"api": "jax.op"},
        },
      },
    }

  def get_definition_by_id(self, op_id):
    return self.data.get(op_id)

  def update_definition(self, op_id, data):
    self.data[op_id] = data


@pytest.fixture
def harvester():
  mgr = MockInferenceSemantics()
  return SemanticHarvester(mgr, target_fw="jax")


def test_harvest_inference_int_axis(harvester, tmp_path):
  """
  Scenario: User calls `sum(arr, dim=1)`.
  Inference:
      - `dim=1`: Literal `1` is `int`.
      - Spec `sum`: `axis` is `int`.
      - Match: `axis` -> `dim`.
  """
  code = """
import jax.numpy as jnp
def test_sum():
    arr = jnp.ones((2,2))
    # 'dim' is target name, 1 is value. Spec expects 'axis: int'.
    res = jnp.sum(arr, dim=1)
"""
  fpath = tmp_path / "test_inf_axis.py"
  fpath.write_text(code)

  harvester.harvest_file(fpath)

  args = harvester.semantics.data["sum"]["variants"]["jax"]["args"]
  assert args.get("axis") == "dim"


def test_harvest_inference_bool_keepdims(harvester, tmp_path):
  """
  Scenario: User calls `mean(arr, keep=True)`.
  Inference:
      - `keep=True`: Literal `True` is `bool`.
      - Spec `mean`: `keepdims` is `bool`.
      - Match: `keepdims` -> `keep`.
  """
  code = """
import jax.numpy as jnp
def test_mean():
    val = jnp.mean(x, keep=True)
"""
  fpath = tmp_path / "test_inf_bool.py"
  fpath.write_text(code)

  harvester.harvest_file(fpath)

  args = harvester.semantics.data["mean"]["variants"]["jax"]["args"]
  assert args.get("keepdims") == "keep"


def test_harvest_ambiguous_inference_skip(harvester, tmp_path):
  """
  Scenario: User calls `op(val_1=1, val_2=2)`.
  Spec: `a: int`, `b: int`.
  Inference:
      - `val_1=1` matches both `a` and `b`.
      - Ambiguity -> Skip inference (return None).
  Result: Heuristic falls back to implicit keyword match (std name = val_1).
  """
  code = """
import jax
def test_ambiguous_op():
    jax.op(val_1=1)
"""
  fpath = tmp_path / "test_ambig.py"
  fpath.write_text(code)

  harvester.harvest_file(fpath)

  # Should default to val_1 -> val_1 because specific mapping logic fails
  # implicitly assume keyword matches std name
  args = harvester.semantics.data["ambiguous_op"]["variants"]["jax"]["args"]
  # The default behavior of TargetCallVisitor is:
  # If inference fails, extracted_map[tgt_arg] = tgt_arg
  # So we expect {"val_1": "val_1"} if "val_1" happens to match std arg name?
  # Actually, extracted_map keys are STD names. If we say extracted_map["val_1"]="val_1",
  # we are saying spec has "val_1".
  # Spec has "a" and "b". So this mapping is technically invalid for the spec,
  # but the harvester persists what it finds.
  assert args.get("val_1") == "val_1"
