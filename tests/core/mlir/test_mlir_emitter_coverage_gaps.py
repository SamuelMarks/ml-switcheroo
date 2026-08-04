"""Module for testing MLIR emitter coverage gaps.

This module provides focused unit tests that target specific edge cases and
coverage gaps in the PythonToMlirEmitter class of ml_switcheroo. Specifically,
it validates the emitter's resilience when handling unresolved base attribute
assignments, while loops, and if-conditional expressions.
"""

import libcst as cst
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter


def test_emitter_assign_unresolved_base() -> None:
  """Tests assign statements with unresolved base references.

  This test covers the scenario (typically line 311) where an attribute assignment
  is made on a base reference that cannot be resolved in the current context
  (e.g., self.unknown.attr = 5). It verifies that parsing and converting this
  source does not crash and produces valid MLIR operations from the assigned value.

  Args:
      None

  Returns:
      None
  """
  # Covers line 311 (unresolved base)
  source = "self.unknown.attr = 5"
  module = cst.parse_module(source)
  emitter = PythonToMlirEmitter()
  node = emitter.convert(module)
  assert len(node.body.operations) > 0  # ops from 5


def test_emitter_assign_unresolved_base2() -> None:
  """Tests assign statements with unresolved base references when base_val is None.

  This test covers an alternative scenario for unresolved base references (line 310)
  where the base lookup results in False or None. It validates that the flat attribute
  extraction correctly returns a string, avoids exceptions, and generates the necessary
  MLIR operations representing the assignment's value.

  Args:
      None

  Returns:
      None
  """
  # Another attempt for 310 if unknown_base was not flattened properly
  source = "self.unknown.attr = 5"
  module = cst.parse_module(source)
  emitter = PythonToMlirEmitter()

  # We want base_val (from lookup(base_name)) to be False/None.
  # self.unknown flattens to "self.unknown".
  # lookup("self.unknown") returns None since it's not in ctx.
  # We need self._flatten_attr(t.value) to return a string (it will: "self.unknown").

  node = emitter.convert(module)
  assert len(node.body.operations) > 0


def test_emitter_emit_while() -> None:
  """Tests emitter behavior on while loops.

  This test covers emitter behavior (typically line 317) when converting code
  containing while loops. It validates that while loop patterns are processed
  successfully, returning an empty list of operations in this minimal loop context.

  Args:
      None

  Returns:
      None
  """
  # Covers line 317
  source = "while True:\n    pass"
  module = cst.parse_module(source)
  emitter = PythonToMlirEmitter()
  node = emitter.convert(module)
  assert len(node.body.operations) == 0


def test_emitter_emit_if() -> None:
  """Tests emitter behavior on conditional if statements.

  This test covers emitter behavior (typically line 321) when converting code
  containing if-statements. It validates that basic conditional blocks are
  processed successfully, returning an empty list of operations in this pass-through
  conditional context.

  Args:
      None

  Returns:
      None
  """
  # Covers line 321
  source = "if True:\n    pass"
  module = cst.parse_module(source)
  emitter = PythonToMlirEmitter()
  node = emitter.convert(module)
  assert len(node.body.operations) == 0
