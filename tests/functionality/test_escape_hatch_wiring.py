"""
Tests for Escape Hatch Wiring in AST Translation.

Verifies that:
1. 'Strict Mode' flags unknown functions (preserves them without crashing).
2. Known APIs with missing Target Variants (Tier C) are preserved.
3. Default mode passes unknown APIs through silently.
"""

import pytest

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """
  Mock Manager injected with specific gap scenarios.
  Overrides __init__ to prevent loading real JSON files during unit test.
  """

  def __init__(self):
    # We skip super().__init__ to avoid loading real JSON files
    self.data = {}
    self.import_data = {}  # Required by ASTEngine since Feature 024
    self.framework_configs = {}  # Required by ASTEngine since Feature 07
    self._reverse_index = {}

    # 1. Tier A: Fully Supported
    self._inject_op(op_name="abs", variants={"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}})

    # 2. Tier C: Source Known, Target Missing (e.g. DataLoader)
    self._inject_op(
      op_name="DataLoader",
      variants={
        "torch": {"api": "torch.utils.data.DataLoader"}
        # Missing JAX key (or None) implies no translation available
      },
    )

  def _inject_op(self, op_name, variants):
    self.data[op_name] = {"variants": variants, "std_args": ["x"]}
    for fw, details in variants.items():
      if "api" in details:
        self._reverse_index[details["api"]] = (op_name, self.data[op_name])


@pytest.fixture
def semantics_mgr():
  return MockSemantics()


def test_escape_hatch_tier_c_gap(semantics_mgr):
  """
  Scenario: The API is 'Known' in PyTorch, but mapped to None/Missing in JAX.
  Expectation: The Rewriter attempts to mark failure.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=False)

  code = "loader = torch.utils.data.DataLoader(ds)"
  result = engine.run(code)

  # 1. Ensure the code was not mangled or deleted
  assert "torch.utils.data.DataLoader(ds)" in result.code

  # 2. Check imports/structure (ImportFixer runs)
  assert "loader =" in result.code

  # Check that error was detected
  assert len(result.errors) >= 1
  # Updated message from ASTEngine.run
  assert "block(s) marked for manual review" in result.errors[0]


def test_strict_mode_unknown_source_api(semantics_mgr):
  """
  Scenario: User calls 'torch.weird_custom_func'.
  Flag: strict_mode = True.
  Expectation: Code is preserved but marked as failure.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)

  code = "y = torch.weird_custom_func(x)"
  result = engine.run(code)

  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is True


def test_strict_mode_ignores_standard_python(semantics_mgr):
  """
  Scenario: User calls 'len(x)' (standard, pure).
  Flag: strict_mode = True.
  Expectation: Standard python is ignored by the Rewriter (no prefix match),
  so no Escape Hatch logic should even be invoked.

  WARNING: Using 'print(x)' causes PurityScanner to trigger a violation side-effect,
  so we use 'len(x)' which is pure.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=True)

  code = "z = len(x)"
  result = engine.run(code)

  # Strictly no markers
  assert EscapeHatch.START_MARKER not in result.code
  assert "z = len(x)" in result.code
  assert result.has_errors is False


def test_default_mode_passthrough(semantics_mgr):
  """
  Scenario: User calls 'torch.weird_custom_func'.
  Flag: strict_mode = False (default).
  Expectation: Silent preservation. Code flows through Rewrite unmodified.
  """
  engine = ASTEngine(semantics=semantics_mgr, source="torch", target="jax", strict_mode=False)

  code = "y = torch.weird_custom_func(x)"
  result = engine.run(code)

  assert EscapeHatch.START_MARKER not in result.code
  assert "torch.weird_custom_func(x)" in result.code
  assert result.has_errors is False
