"""Import generation strategies for Python backend synthesis."""

import libcst as cst

from ml_switcheroo.frameworks.base import get_adapter


def generate_imports(framework: str) -> list[cst.SimpleStatementLine]:
  """Generate standard top-level import CST statements for the target framework.

  Reads ``default_imports`` from the registered ``FrameworkAdapter``'s
  ``StructuralTraits`` configuration.

  Args:
      framework: Target framework identifier (e.g. 'torch', 'jax', 'mlx').

  Returns:
      List of parsed LibCST SimpleStatementLine nodes for the target framework.
  """
  adapter = get_adapter(framework)
  stmt_strs = adapter.structural_traits.default_imports if adapter else []
  out: list[cst.SimpleStatementLine] = []
  for s in stmt_strs:
    stmt = cst.parse_statement(s)
    assert isinstance(stmt, cst.SimpleStatementLine)
    out.append(stmt)
  return out
