"""Test suite for the Purity module."""

import libcst as cst
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics:
  """Mock Semantics class for testing purposes."""

  def get_all_rng_methods(self):
    """Mock implementation of get all rng methods."""
    return {"custom_seed"}

  def get_framework_config(self, framework):
    """Mock implementation of get framework configuration."""
    if framework == "torch":
      return {"traits": {"impurity_methods": ["add_", "copy_"]}}
    return {}


def analyze(code: str, use_semantics: bool = False) -> str:
  """Analyzes .

  Args:
      code: ...
      use_semantics: ...
  """
  semantics = MockSemantics() if use_semantics else None
  tree = cst.parse_module(code)
  scanner = PurityScanner(semantics=semantics, source_fw="torch")
  new_tree = tree.visit(scanner)
  return new_tree.code


def test_io_detection_print():
  """Verifies the behavior of I/O detection print."""
  code = "print(x)"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "Side-effect unsafe for JAX: I/O Call (print)" in result


def test_mutation_detection_list_append():
  """Verifies the behavior of mutation detection list append."""
  code = "my_list.append(item)"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "In-place Mutation (. append)" in result


def test_global_keyword_detection():
  """Verifies the behavior of global keyword detection."""
  code = "\ndef f():\n    global x\n    x = 1\n"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "Global mutation (x)" in result


def test_nonlocal_keyword_detection():
  """Verifies the behavior of nonlocal keyword detection."""
  code = "\ndef outer():\n    x = 0\n    def inner():\n        nonlocal x\n        x = 1\n"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "Nonlocal mutation (x)" in result


def test_rng_seed_detection_dynamic():
  """Verifies the behavior of rng seed detection dynamic."""
  code = "lib.custom_seed(123)"
  result = analyze(code, use_semantics=True)
  assert EscapeHatch.START_MARKER in result
  assert "Global RNG State (. custom_seed)" in result


def test_framework_specific_impurity():
  """Verifies the behavior of framework specific impurity."""
  code = "x.add_(y)"
  result = analyze(code, use_semantics=True)
  assert EscapeHatch.START_MARKER in result
  assert "State Mutation (. add_)" in result


def test_pure_code_passes_clean():
  """Verifies the behavior of pure code passes clean."""
  code = "def add(x, y): return x + y"
  result = analyze(code)
  assert EscapeHatch.START_MARKER not in result
  assert result.strip() == code.strip()


def test_file_write_detection():
  """Verifies the behavior of file write detection."""
  code = "f.write('data')"
  result = analyze(code)
  assert EscapeHatch.START_MARKER in result
  assert "I/O Call (.write)" in result


class IncompleteSemantics:
  """Test suite for the Incomplete Semantics component."""

  def get_framework_config(self, framework):
    """Gets framework configuration."""
    return {"other": "stuff"}


def test_purity_missing_branches():
  """Verifies the behavior of purity missing branches."""
  semantics = IncompleteSemantics()
  _scanner = PurityScanner(semantics=semantics, source_fw="torch")
  _scanner2 = PurityScanner(semantics=MockSemantics(), source_fw=None)
  code = "\nmy_func()  # Call with Name, but not I/O\nmy_list.copy()  # Call with Attribute, but not mutating\n(lambda x: x)(1)  # Call with func that is not Name or Attribute\n"
  result = analyze(code, use_semantics=False)
  assert EscapeHatch.START_MARKER not in result
