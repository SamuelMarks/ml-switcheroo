"""
Tests for Fuzzy Matching and Signature Analysis in Scaffolder.

Verifies that:
1.  Fuzzy name matching connects synonyms (absolute -> abs).
2.  **Signature Analysis** prevents false positives (e.g. unary `foo` vs multi-arg `foo`).
3.  Exact matches are prioritized.
"""

import json
import pytest
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector(ApiInspector):
  """Overrides inspect to return deterministic fake catalogs."""

  def inspect(self, fw_name: str) -> dict:
    if fw_name == "source_fw":
      return {
        "source.absolute": {
          "name": "absolute",
          "params": ["x"],
          "docstring_summary": "Abs value",
        },
        "source.add": {
          "name": "add",
          "params": ["x", "y"],
          "docstring_summary": "Add",
        },
        "source.unary_op": {
          "name": "unary_op",
          "params": ["x"],
          "docstring_summary": "Takes one arg",
        },
      }

    if fw_name == "target_fw":
      return {
        "target.abs": {
          # MATCH MATCH: 'abs' ~ 'absolute' + Arity(1) == Arity(1)
          "name": "abs",
          "params": ["a"],
          "docstring_summary": "Abs",
        },
        "target.add": {
          # EXACT MATCH
          "name": "add",
          "params": ["a", "b"],
          "docstring_summary": "Add",
        },
        "target.wrong_arity_op": {
          # NAME MATCH but ARITY MISMATCH
          # unary_op vs this (3 params)
          "name": "unary_op",
          "params": ["a", "b", "c"],
          "docstring_summary": "Same name, different logic",
        },
      }
    return {}


@pytest.fixture
def clean_semantics():
  """Returns a manager with no pre-loaded data to ensure tests rely on heuristics."""
  mgr = SemanticsManager()
  mgr.data = {}
  mgr._key_origins = {}
  return mgr


def test_fuzzy_match_success(tmp_path, clean_semantics):
  """
  Verify that 'absolute' (Source) matches 'abs' (Target)
  because names are similar AND arity matches (1 param vs 1 param).
  """
  scaffolder = Scaffolder(semantics=clean_semantics, similarity_threshold=0.6)
  scaffolder.inspector = MockInspector()

  scaffolder.scaffold(["source_fw", "target_fw"], tmp_path)

  out_file = tmp_path / "k_array_api.json"
  assert out_file.exists()

  with open(out_file, "rt", encoding="utf-8") as f:
    data = json.load(f)

  # Check 'absolute' entry
  assert "absolute" in data
  variants = data["absolute"]["variants"]

  # Target should be matched fuzzily
  assert "target_fw" in variants
  assert variants["target_fw"]["api"] == "target.abs"


def test_signature_analysis_rejection(tmp_path, clean_semantics):
  """
  Verify that signature mismatch prevents linking even if names are identical.

  Scenario: Source `unary_op(x)` vs Target `unary_op(a, b, c)`.
  Expectation: The target variant is NOT added due to arity penalty.
  """
  scaffolder = Scaffolder(semantics=clean_semantics, similarity_threshold=0.8, arity_penalty=0.5)
  scaffolder.inspector = MockInspector()

  scaffolder.scaffold(["source_fw", "target_fw"], tmp_path)

  out_file = tmp_path / "k_array_api.json"
  with open(out_file, "rt", encoding="utf-8") as f:
    data = json.load(f)

  assert "unary_op" in data
  variants = data["unary_op"]["variants"]

  # Source exists
  assert "source_fw" in variants
  # Target should be REJECTED despite name match 'unary_op' == 'unary_op'
  assert "target_fw" not in variants


def test_exact_match_priority(tmp_path, clean_semantics):
  """
  Verify that if an exact name exists with correct arity, it is accepted.
  """
  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockInspector()

  scaffolder.scaffold(["source_fw", "target_fw"], tmp_path)

  out_file = tmp_path / "k_array_api.json"
  with open(out_file, "rt", encoding="utf-8") as f:
    data = json.load(f)

  # 'add' matches 'add'
  assert "add" in data
  assert data["add"]["variants"]["target_fw"]["api"] == "target.add"
