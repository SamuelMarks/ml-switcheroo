"""
Tests for Purity Analysis (JAX Safety Checks).

Verifies Feature 051 & Global/Nonlocal Detection:
1.  Detection of I/O (`print`, `input`).
2.  Detection of Global state modification.
3.  Detection of Nonlocal (closure) state modification.
4.  Detection of In-place Mutation (`append`, `extend`).
5.  Detection of Global RNG Seeding (`random.seed`, `torch.manual_seed`).
6.  Correct wrapping via EscapeHatch.
"""

import libcst as cst
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.core.escape_hatch import EscapeHatch


def analyze(code: str) -> str:
  """
  Helper to parse, scan, and re-emit code.

  Args:
      code: The source string to analyze.

  Returns:
      The source string potentially wrapped in escape hatches.
  """
  tree = cst.parse_module(code)
  scanner = PurityScanner()
  new_tree = tree.visit(scanner)
  return new_tree.code


def test_io_detection_print():
  """
  Verification: 'print(x)' is caught as I/O side effect.
  """
  code = "print(x)"
  result = analyze(code)

  assert EscapeHatch.START_MARKER in result
  assert "Side-effect unsafe for JAX: I/O Call (print)" in result
  assert "print(x)" in result  # Code preserved inside marker


def test_io_detection_input():
  """
  Verification: 'input()' is caught.
  """
  code = "val = input('Prompt:')"
  result = analyze(code)

  assert EscapeHatch.START_MARKER in result
  assert "I/O Call (input)" in result


def test_mutation_detection_list_append():
  """
  Verification: '.append()' is caught as in-place mutation.
  """
  code = "my_list.append(item)"
  result = analyze(code)

  assert EscapeHatch.START_MARKER in result
  assert "In-place Mutation (. append)" in result


def test_mutation_detection_list_extend_pop():
  """
  Verification: multiple mutation methods (.extend, .pop) are caught.
  """
  code = """
data.extend(new_data)
val = data.pop()
"""
  result = analyze(code)

  # Both lines should be wrapped
  assert result.count(EscapeHatch.START_MARKER) == 2
  assert "In-place Mutation (. extend)" in result
  assert "In-place Mutation (. pop)" in result


def test_global_keyword_detection():
  """
  Verification: 'global x' is flagged.
  """
  code = """
def func():
    global x
    x = 10
"""
  result = analyze(code)

  assert EscapeHatch.START_MARKER in result
  assert "Global mutation (x)" in result


def test_nonlocal_keyword_detection():
  """
  Verification: 'nonlocal x' is flagged.
  """
  code = """
def outer():
    x = 0
    def inner():
        nonlocal x
        x = 1
"""
  result = analyze(code)

  assert EscapeHatch.START_MARKER in result
  assert "Nonlocal mutation (x)" in result


def test_rng_seed_detection():
  """
  Verification: Random seeding is caught as Global State mutation.
  Covers: random.seed, numpy.random.seed, torch.manual_seed.
  """
  code = """
import random
import torch
import numpy as np

random.seed(42)
np.random.seed(123)
torch.manual_seed(0)
"""
  result = analyze(code)

  # All three seeding calls should be marked
  assert result.count(EscapeHatch.START_MARKER) == 3
  # Check for specific reasons
  assert "Global RNG State (. seed)" in result
  assert "Global RNG State (. manual_seed)" in result


def test_pure_code_passes_clean():
  """
  Verification: Pure mathematical code is not flagged.
  """
  code = """
def add(x, y):
    return x + y
"""
  result = analyze(code)

  assert EscapeHatch.START_MARKER not in result
  assert code.strip() in result


def test_file_write_detection():
  """
  Verification: .write() is caught (generic attribute check).
  """
  code = "f.write('data')"
  result = analyze(code)

  assert EscapeHatch.START_MARKER in result
  assert "I/O Call (.write)" in result


def test_multiple_violations_single_marker():
  """
  Verification: multiple violations in one statement result in one marker block
  with combined reasons.
  """
  # Statement: print(l.pop()) -> Both I/O and Mutation
  code = "print(l.pop())"
  result = analyze(code)

  # Should wrap once
  assert result.count(EscapeHatch.START_MARKER) == 1

  # Reason should contain both
  assert "I/O Call (print)" in result
  assert "In-place Mutation (. pop)" in result
