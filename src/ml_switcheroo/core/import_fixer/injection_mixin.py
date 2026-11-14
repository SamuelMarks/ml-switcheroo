"""Module-Level Import Injector Collaborator.

Handles post-processing of ``cst.Module`` nodes to inject missing top-level
target imports required by ``ResolutionPlan`` and deduplicate import lines.
"""

import libcst as cst

from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.utils import create_dotted_name, get_signature, is_docstring, is_future_import


class ImportInjector:
  """Collaborator injecting unsatisfied top-level imports into ``cst.Module`` bodies.

  Attributes:
      state: Shared ``ImportFixerState`` instance.
  """

  def __init__(self, state: ImportFixerState) -> None:
    """Initialize the module import injector with shared ``ImportFixerState``.

    Args:
        state: Shared ``ImportFixerState`` instance.
    """
    self.state: ImportFixerState = state

  @staticmethod
  def _deduplicate_imports(
    merged_body: list[cst.SimpleStatementLine | cst.BaseCompoundStatement],
  ) -> list[cst.SimpleStatementLine | cst.BaseCompoundStatement]:
    """Deduplicate pure import statements in ``merged_body`` by normalized signature.

    Args:
        merged_body: Combined module statement list after insertion.

    Returns:
        Deduplicated module statement list.
    """
    clean_body: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
    seen_imports: set[str] = set()
    for stmt in merged_body:
      if (
        isinstance(stmt, cst.SimpleStatementLine)
        and bool(stmt.body)
        and all(isinstance(small, (cst.Import, cst.ImportFrom)) for small in stmt.body)
      ):
        sig = get_signature(stmt)
        if sig in seen_imports:
          continue
        seen_imports.add(sig)
      clean_body.append(stmt)
    return clean_body

  def inject_module_imports(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """Inject required missing imports and deduplicate import statements in ``updated_node``.

    Args:
        original_node: Original ``cst.Module`` node prior to transformation.
        updated_node: Updated ``cst.Module`` node after child statement rewriting.

    Returns:
        Modified ``cst.Module`` containing injected and deduplicated imports.
    """
    injections: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []

    for req in self.state.plan.required_imports:
      if req.signature in self.state.satisfied_injections:
        continue

      check_name = req.alias if req.alias else (req.subcomponent if req.subcomponent else req.module.split(".")[0])
      if check_name in self.state.defined_names:
        continue

      nm = f"{req.module}.{req.subcomponent}" if req.subcomponent else req.module
      leaf = nm.split(".")[-1]
      asname_node = cst.AsName(name=cst.Name(req.alias)) if req.alias and (req.alias != leaf or "." in nm) else None

      node = cst.SimpleStatementLine(
        body=[cst.Import(names=[cst.ImportAlias(name=create_dotted_name(nm), asname=asname_node)])]
      )
      self.append_injection(injections, node)
      self.state.defined_names.add(check_name)
      self.state.satisfied_injections.add(req.signature)

    body_stats = list(updated_node.body)
    insert_idx = 0
    for i, stmt in enumerate(body_stats):
      if is_docstring(stmt, i) or is_future_import(stmt):
        insert_idx = i + 1
        continue
      break

    merged_body = body_stats[:insert_idx] + injections + body_stats[insert_idx:]
    return updated_node.with_changes(body=self._deduplicate_imports(merged_body))

  def append_injection(
    self,
    injections_list: list[cst.SimpleStatementLine | cst.BaseCompoundStatement],
    node: cst.SimpleStatementLine,
  ) -> None:
    """Append an import statement CST node to the pending module-level injection list.

    Args:
        injections_list: Pending list of statement CST nodes to inject.
        node: Import ``SimpleStatementLine`` node to append.
    """
    injections_list.append(node)


__all__ = ["ImportInjector"]
