"""Test module."""

import libcst as cst
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.semantics.manager import SemanticsManager


def get_hatch_message(tree: cst.Module) -> str:
  """Test element."""
  # We look for the escape hatch comment
  lines = tree.code.splitlines()
  for line in lines:
    if line.startswith("# Reason: Side-effect unsafe for JAX"):
      return line
  return ""


def test_purity_io():
  """Test element."""
  scanner = PurityScanner()
  code = "print('hello')"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "Side-effect unsafe for JAX" in msg
  assert "I/O Call (print)" in msg


def test_purity_mutation():
  """Test element."""
  scanner = PurityScanner()
  code = "x.append(1)"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "In-place Mutation (. append)" in msg


def test_purity_io_write():
  """Test element."""
  scanner = PurityScanner()
  code = "f.write('hello')"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "I/O Call (.write)" in msg


def test_purity_global():
  """Test element."""
  scanner = PurityScanner()
  code = "global x"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "Global mutation (x)" in msg


def test_purity_nonlocal():
  """Test element."""
  scanner = PurityScanner()
  code = "nonlocal y, z"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "Nonlocal mutation (y, z)" in msg


def test_purity_rng():
  """Test element."""
  scanner = PurityScanner()
  code = "random.seed(42)"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "Global RNG State (. seed)" in msg


def test_purity_dynamic_config():
  """Test element."""
  semantics = SemanticsManager()
  # Mock framework config
  semantics.framework_configs["torch"] = {"traits": {"impurity_methods": ["add_", "copy_"]}}

  scanner = PurityScanner(semantics=semantics, source_fw="torch")
  code = "x.add_(1)"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "State Mutation (. add_)" in msg


def test_purity_multiple_violations():
  """Test element."""
  scanner = PurityScanner()
  code = "print(x.append(1))"  # Contrived, but shows multiple
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert "I/O Call (print)" in msg
  assert "In-place Mutation (. append)" in msg


def test_purity_safe():
  """Test element."""
  scanner = PurityScanner()
  code = "x = y + z"
  tree = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg = get_hatch_message(tree)
  assert msg == ""
