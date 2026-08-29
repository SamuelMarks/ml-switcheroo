"""Test module."""

import libcst as cst

from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.semantics.manager import SemanticsManager


def get_hatch_message(tree: cst.Module) -> str:
  """Docstring."""
  # We look for the escape hatch comment
  lines: list[str] = tree.code.splitlines()
  line: str
  for line in lines:
    if line.startswith("# Reason: Side-effect unsafe for JAX"):
      return line
  return ""


def test_purity_io() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "print('hello')"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Side-effect unsafe for JAX" in msg
  assert "I/O Call (print)" in msg


def test_purity_mutation() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "x.append(1)"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "In-place Mutation (. append)" in msg


def test_purity_io_write() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "f.write('hello')"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "I/O Call (.write)" in msg


def test_purity_global() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "global x"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Global mutation (x)" in msg


def test_purity_nonlocal() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "nonlocal y, z"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Nonlocal mutation (y, z)" in msg


def test_purity_rng() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "random.seed(42)"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Global RNG State (. seed)" in msg


def test_purity_dynamic_config() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  # Mock framework config
  semantics.framework_configs["torch"] = {"traits": {"impurity_methods": ["add_", "copy_"]}}

  scanner: PurityScanner = PurityScanner(semantics=semantics, source_fw="torch")
  code: str = "x.add_(1)"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "State Mutation (. add_)" in msg


def test_purity_multiple_violations() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "print(x.append(1))"  # Contrived, but shows multiple
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "I/O Call (print)" in msg
  assert "In-place Mutation (. append)" in msg


def test_purity_safe() -> None:
  """Docstring."""
  scanner: PurityScanner = PurityScanner()
  code: str = "x = y + z"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert msg == ""
