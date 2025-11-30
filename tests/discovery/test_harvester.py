"""
Tests for Semantic Harvester.

Verifies that the harvester correctly:
1. Parses manual test files.
2. Identifies relevant test functions.
3. Extracts argument mappings from API calls.
4. Handles import aliases.
5. Updates the SemanticsManager.
"""

import pytest
from unittest.mock import MagicMock

from ml_switcheroo.discovery.harvester import SemanticHarvester
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics():
  """Returns a mocked SemanticsManager with stubbed methods."""
  mgr = MagicMock(spec=SemanticsManager)

  # Default behavior: define 'add' as known, but without specific args
  mgr.get_definition_by_id.return_value = {"variants": {"jax": {"api": "jax.numpy.add", "args": {}}}}
  return mgr


@pytest.fixture
def harvester(mock_semantics):
  """Returns a harvester targeted at JAX."""
  return SemanticHarvester(mock_semantics, target_fw="jax")


def test_harvest_keyword_args(harvester, tmp_path):
  """
  Scenario: Manual test uses explicit keywords.
  Statement: `jnp.add(x=1, alpha=0.5)`
  Logic:
      - Function call `jnp.add` matches target `jax.numpy.add` (via alias check).
      - Keyword `x` -> map to std `x` (assumed literal match).
      - Keyword `alpha` -> map to std `alpha` (assumed literal match).
  Result: Arg maps updated to include these keys.
  """
  test_code = """
import jax.numpy as jnp
def test_add():
    # Calling target framework with corrective keywords
    res = jnp.add(x=1, alpha=0.5)
"""
  fpath = tmp_path / "test_manual.py"
  fpath.write_text(test_code)

  # Run
  count = harvester.harvest_file(fpath)

  # Verify
  assert count == 1

  # Check that update_definition was called
  harvester.semantics.update_definition.assert_called_once()

  call_args = harvester.semantics.update_definition.call_args
  op_name = call_args[0][0]
  data = call_args[0][1]

  assert op_name == "add"
  # Note: TargetCallVisitor uses literal kwarg Name as std_arg if value isn't a variable
  mapped_args = data["variants"]["jax"]["args"]
  assert mapped_args["x"] == "x"
  assert mapped_args["alpha"] == "alpha"


def test_harvest_variable_naming_convention(harvester, tmp_path):
  """
  Scenario: Manual test uses `np_<std>` convention.
  Statement: `jax.numpy.sum(a=np_x, axis=np_axis)`
  Logic:
      - `a=np_x`: target='a', value='np_x' -> std='x'. Map: 'x' -> 'a'.
      - `axis=np_axis`: target='axis', value='np_axis' -> std='axis'. Map: 'axis' -> 'axis'.
  """
  # Setup mock for 'sum'
  harvester.semantics.get_definition_by_id.side_effect = (
    lambda op: {"variants": {"jax": {"api": "jax.numpy.sum", "args": {}}}} if op == "sum" else None
  )

  test_code = """
import jax
def test_sum():
    np_x = create_tensor()
    np_axis = 1
    # Specific Mapping logic
    res = jax.numpy.sum(a=np_x, axis=np_axis)
"""
  fpath = tmp_path / "test_mapping.py"
  fpath.write_text(test_code)

  harvester.harvest_file(fpath)

  call_args = harvester.semantics.update_definition.call_args[0]
  op_name = call_args[0]
  new_args = call_args[1]["variants"]["jax"]["args"]

  assert op_name == "sum"
  assert new_args["x"] == "a"
  assert new_args["axis"] == "axis"


def test_harvest_ignores_irrelevant_functions(harvester, tmp_path):
  """
  Scenario: File has `test_other` calling `jax.numpy.sum`.
  Logic:
      - `test_other` -> implies op `other`.
      - Semantics for `other` (mocked) matches nothing or `sum` call is inside.
      - The visitor only extracts if the call name matches the op's target API.
      - `other` is not `sum`, so `jax.numpy.sum` call should be ignored.
  """

  # Semantics returns define for 'other' but points to 'jax.numpy.other'
  def side_effect(op):
    if op == "other":
      return {"variants": {"jax": {"api": "jax.numpy.other"}}}
    return None

  harvester.semantics.get_definition_by_id.side_effect = side_effect

  test_code = """
def test_other():
    # This call shouldn't be harvested because function tested is 'other', 
    # but call is 'sum'.
    jax.numpy.sum(a=1)
"""
  fpath = tmp_path / "test_ignore.py"
  fpath.write_text(test_code)

  count = harvester.harvest_file(fpath)
  assert count == 0


def test_harvest_multiple_in_file(harvester, tmp_path):
  """
  Scenario: Two tests in one file (`test_add`, `test_sub`).
  Verify both are processed.
  """
  test_code = """
import jax.numpy as jnp

def test_add():
    jnp.add(x=1)

def test_sub():
    jnp.subtract(x=1)
"""
  fpath = tmp_path / "test_multi.py"
  fpath.write_text(test_code)

  # Mock return for both ops
  def side_effect(op_id):
    if op_id == "add":
      return {"variants": {"jax": {"api": "jax.numpy.add", "args": {}}}}
    if op_id == "sub":
      return {"variants": {"jax": {"api": "jax.numpy.subtract", "args": {}}}}
    return None

  harvester.semantics.get_definition_by_id.side_effect = side_effect

  count = harvester.harvest_file(fpath)
  assert count == 2


def test_harvest_dry_run(harvester, tmp_path):
  """
  Verify dry_run does not call update_definition.
  """
  test_code = "def test_add(): jax.numpy.add(x=1)"
  fpath = tmp_path / "dry_run.py"
  fpath.write_text(test_code)

  harvester.harvest_file(fpath, dry_run=True)

  harvester.semantics.update_definition.assert_not_called()
