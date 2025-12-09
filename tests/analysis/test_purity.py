"""
Tests for Purity Analysis (JAX Safety Checks).

Verifies Feature 051 & Global/Nonlocal Detection:
1.  Detection of I/O (`print`, `input`).
2.  Detection of Global state modification.
3.  Detection of Nonlocal state modification.
4.  Detection of standard In-place Mutation (`append`, `extend`).
5.  Detection of Global RNG Seeding (`random.seed`).
6.  Detection of **Framework Specific Impurities** (`add_`).
"""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics:
  def get_all_rng_methods(self):
    return {"custom_seed"}

  def get_framework_config(self, framework):
    if framework == "torch":
      return {"traits": {"impurity_methods": ["add_", "copy_"]}}
    return {}


def analyze(code: str, use_semantics: bool = False) -> str:
  """
  Helper to parse, scan, and re-emit code.
  """
  semantics = MockSemantics() if use_semantics else None

  tree = cst.parse_module(code)
  scanner = PurityScanner(semantics=semantics, source_fw="torch")
  new_tree = tree.visit(scanner)
  return new_tree.code


def test_io_detection_print():
  code = "print(x)"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "Side-effect unsafe for JAX: I/O Call (print)" in result


def test_mutation_detection_list_append():
  code = "my_list.append(item)"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "In-place Mutation (. append)" in result


def test_global_keyword_detection():
  # Use multiline string to ensure SimpleStatementLine wrapping occurs
  code = """
def f():
    global x
    x = 1
"""
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "Global mutation (x)" in result


def test_nonlocal_keyword_detection():
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


def test_rng_seed_detection_dynamic():
  """
  Verification: Dynamic seed methods from SemanticsManager are caught.
  """
  code = "lib.custom_seed(123)"
  result = analyze(code, use_semantics=True)
  assert EscapeHatch.START_MARKER in result
  assert "Global RNG State (. custom_seed)" in result


def test_framework_specific_impurity():
  """
  Verification: Framework-specific mutation methods (add_) defined in schema are caught.
  """
  code = "x.add_(y)"
  result = analyze(code, use_semantics=True)

  # Should catch 'add_' because MockSemantics provides it for 'torch' source
  assert EscapeHatch.START_MARKER in result
  assert "State Mutation (. add_)" in result


def test_pure_code_passes_clean():
  code = "def add(x, y): return x + y"
  result = analyze(code)
  assert EscapeHatch.START_MARKER not in result
  assert result.strip() == code.strip()


def test_file_write_detection():
  code = "f.write('data')"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "I/O Call (.write)" in result
