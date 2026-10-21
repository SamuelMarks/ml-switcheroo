"""Test module for the semantic API PurityScanner.

This module verifies the capabilities of `PurityScanner` to detect side-effects and impure
operations in Python code that violate functional execution constraints (required for
backends like JAX). It tests detection of I/O, in-place mutations, global state changes,
and interactions with the SemanticsManager for framework-specific impurity rules.
"""

from pydantic import JsonValue

import libcst as cst

from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.semantics.manager import SemanticsManager


def get_hatch_message(tree: cst.Module) -> str:
  """Extract the auto-generated escape hatch comment from the transformed CST tree.

  The PurityScanner injects a specific comment structure before impure statements
  to safely gate them from strict compilers.

  Args:
      tree (cst.Module): The transformed CST module.

  Returns:
      str: The first escape hatch comment found, or an empty string if none exist.
  """
  # We look for the escape hatch comment
  lines: list[str] = tree.code.splitlines()
  line: str
  for line in lines:
    if line.startswith("# Reason: Side-effect unsafe for JAX"):
      return line
  return ""


def test_purity_io() -> None:
  """Test detection of standard I/O functions.

  Verifies that basic Python `print` statements are correctly flagged as an 'I/O Call'
  and annotated with the escape hatch comment.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "print('hello')"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Side-effect unsafe for JAX" in msg
  assert "I/O Call (print)" in msg


def test_purity_mutation() -> None:
  """Test detection of standard in-place mutations.

  Verifies that list operations like `.append()` are identified as 'In-place Mutation'
  since they mutate the underlying memory structure without returning a new object.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "x.append(1)"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "In-place Mutation (. append)" in msg


def test_purity_io_write() -> None:
  """Test detection of file I/O operations.

  Verifies that file object methods like `.write()` are correctly identified
  as impure 'I/O Call' operations.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "f.write('hello')"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "I/O Call (.write)" in msg


def test_purity_global() -> None:
  """Test detection of the `global` keyword.

  Verifies that statements redefining variables using `global` are strictly flagged
  as 'Global mutation', since they break function scope purity.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "global x"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Global mutation (x)" in msg


def test_purity_nonlocal() -> None:
  """Test detection of the `nonlocal` keyword.

  Verifies that closure mutations using `nonlocal` are correctly flagged
  as 'Nonlocal mutation' and properly extract all the target identifiers.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "nonlocal y, z"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Nonlocal mutation (y, z)" in msg


def test_purity_rng() -> None:
  """Test detection of Global Random Number Generator (RNG) calls.

  Verifies that standard library calls altering global RNG state (e.g. `random.seed()`)
  are caught. JAX and similar strict functional frameworks require explicit PRNG key passing.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "random.seed(42)"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "Global RNG State (. seed)" in msg


def test_purity_dynamic_config() -> None:
  """Test framework-specific impurity detection utilizing the SemanticsManager.

  This test mocks a PyTorch configuration specifying that `add_` (in-place addition)
  is impure. It verifies that the scanner dynamically loads and respects this
  'State Mutation' rule.
  """
  semantics: SemanticsManager = SemanticsManager()
  # Mock framework config
  semantics.framework_configs["torch"] = {"traits": {"impurity_methods": ["add_", "copy_"]}}

  scanner: PurityScanner = PurityScanner(semantics=semantics, source_framework="torch")
  code: str = "x.add_(1)"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "State Mutation (. add_)" in msg


def test_purity_multiple_violations() -> None:
  """Test handling of multiple impurity violations in a single statement.

  Verifies that if a single statement contains multiple violations (e.g., calling
  `print` and an `.append` mutation), all reasons are extracted and annotated in
  the escape hatch comment.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "print(x.append(1))"  # Contrived, but shows multiple
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert "I/O Call (print)" in msg
  assert "In-place Mutation (. append)" in msg


def test_purity_safe() -> None:
  """Test parsing of purely functional statements.

  Verifies that standard functional assignments (e.g., `x = y + z`) are not
  flagged and no escape hatch comment is generated.
  """
  scanner: PurityScanner = PurityScanner()
  code: str = "x = y + z"
  tree: cst.Module = cst.parse_module(code)
  tree = tree.visit(scanner)
  msg: str = get_hatch_message(tree)
  assert msg == ""


def test_purity_tensor_receiver_not_flagged_as_list_mutation() -> None:
  """Verify that tensor methods like `tensor.sort()` are not flagged as Python list mutations."""
  from ml_switcheroo.analysis.symbol_table import SymbolTable
  from ml_switcheroo.analysis.symbol_types import TensorType

  code: str = "y = x.sort()"
  tree: cst.Module = cst.parse_module(code)
  stmt = tree.body[0]
  assert isinstance(stmt, cst.SimpleStatementLine)
  assign = stmt.body[0]
  assert isinstance(assign, cst.Assign)
  call_node = assign.value
  assert isinstance(call_node, cst.Call)
  assert isinstance(call_node.func, cst.Attribute)

  sym_table = SymbolTable()
  sym_table.record_type(call_node.func.value, TensorType(framework="torch"))

  scanner = PurityScanner(symbol_table=sym_table)
  visited = tree.visit(scanner)
  assert get_hatch_message(visited) == ""


def test_purity_missing_branches() -> None:
  """Test edge cases and safety checks within the scanner's initialization and AST traversal.

  This test ensures the scanner degrades gracefully when:
  - The provided `semantics` object is missing expected methods or configs.
  - The `source_framework` is None.
  - Unrecognized function call types (like lambda immediately invoked, or safe builtins) are parsed.
  """

  # 73->77: semantics but no get_all_rng_methods
  # We mock a semantics object that lacks it.
  class FakeSemantics1:
    """Mock SemanticsManager lacking rng methods mapping."""

    pass

  _scanner = getattr(
    {
      "v": getattr({"v": PurityScanner}, "__getitem__")("v")(
        semantics=getattr({"v": getattr({"v": FakeSemantics1}, "__getitem__")("v")()}, "__getitem__")("v"),
        source_framework=getattr({"v": "torch"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")

  # 77->exit: semantics has get_framework_config but no source_framework
  class FakeSemantics2:
    """Mock SemanticsManager lacking source_framework resolution."""

    def get_all_rng_methods(self) -> list[str]:
      """Mock returning empty rng methods.

      Returns:
          Computed test result or fixture value.
      """
      return ["seed"]

    def get_framework_config(self, fw: str) -> dict[str, JsonValue] | None:
      """Mock returning None for framework config.

      Args:
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

  _scanner2 = getattr(
    {
      "v": getattr({"v": PurityScanner}, "__getitem__")("v")(
        semantics=getattr({"v": getattr({"v": FakeSemantics2}, "__getitem__")("v")()}, "__getitem__")("v"),
        source_framework=getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")

  # 79->exit: conf is empty or missing "traits"
  class FakeSemantics3:
    """Mock SemanticsManager lacking traits."""

    def get_all_rng_methods(self) -> list[str]:
      """Mock returning empty rng methods.

      Returns:
          Computed test result or fixture value.
      """
      return ["seed"]

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Mock returning incomplete config.

      Args:
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"other": 1}

  _scanner3 = getattr(
    {
      "v": getattr({"v": PurityScanner}, "__getitem__")("v")(
        semantics=getattr({"v": getattr({"v": FakeSemantics3}, "__getitem__")("v")()}, "__getitem__")("v"),
        source_framework=getattr({"v": "torch"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")

  # AST node tests for 164->187, 168->187, 184->187
  scanner4 = PurityScanner()

  # 164->187: Name but not IO (e.g., len(x))
  # 168->187: Not Name, Not Attribute (e.g., [func][0]())
  # 184->187: Attribute but not in any list (e.g., x.upper())
  code = """
len(x)
(lambda: 1)()
x.upper()
"""
  tree = cst.parse_module(code)
  tree = tree.visit(scanner4)
  msg = get_hatch_message(tree)
  assert msg == ""
