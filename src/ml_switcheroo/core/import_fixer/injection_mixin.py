"""
Import Injection Mixin.

Handles the post-processing of the Module AST to inject necessary top-level imports
determined by the `ResolutionPlan`.
"""

from typing import List, Set

import libcst as cst

from ml_switcheroo.core.import_fixer.utils import (
  create_dotted_name,
  get_signature,
  is_docstring,
  is_future_import,
)


class InjectionMixin(cst.CSTTransformer):
  """
  Mixin for injecting imports at the Module level.
  """

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Post-process module to inject imports from the plan.
    """
    injections: List[cst.CSTNode] = []

    for req in self.plan.required_imports:
      if req.signature in self._satisfied_injections:
        continue

      check_name = req.alias if req.alias else (req.subcomponent if req.subcomponent else req.module.split(".")[0])
      if check_name in self._defined_names:
        continue

      # Logic:
      # Requirement: module='mlx', sub='nn', alias='nn'
      # Desired: 'import mlx.nn as nn'

      nm = f"{req.module}.{req.subcomponent}" if req.subcomponent else req.module
      asname_node = None

      # Redundancy check
      should_alias = False
      leaf = nm.split(".")[-1]

      if req.alias:
        if req.alias != leaf:
          should_alias = True
        elif "." in nm:
          # e.g. import torch.nn as nn
          should_alias = True

      if should_alias and req.alias:
        asname_node = cst.AsName(name=cst.Name(req.alias))

      # We exclusively use `import X.Y as Z` style for robustness
      # unless specifically needing `from`. The codebase tests expect
      # `import mlx.nn as nn` and `import torch.nn as nn`.

      node = cst.SimpleStatementLine(
        body=[cst.Import(names=[cst.ImportAlias(name=create_dotted_name(nm), asname=asname_node)])]
      )

      self._append_injection(injections, node)
      self._defined_names.add(check_name)
      self._satisfied_injections.add(req.signature)

    # Insertion Logic
    body_stats = list(updated_node.body)
    insert_idx = 0

    for i, stmt in enumerate(body_stats):
      if is_docstring(stmt, i) or is_future_import(stmt):
        insert_idx = i + 1
        continue
      break

    merged_body = body_stats[:insert_idx] + injections + body_stats[insert_idx:]

    # Deduplication
    clean_body = []
    seen_imports: Set[str] = set()

    for stmt in merged_body:
      is_import = False
      sig = ""

      if isinstance(stmt, cst.SimpleStatementLine):
        all_imports = True
        for small in stmt.body:
          if not isinstance(small, (cst.Import, cst.ImportFrom)):
            all_imports = False
            break
        if all_imports and stmt.body:
          is_import = True
          sig = get_signature(stmt)
      if is_import:
        if sig in seen_imports:
          continue
        seen_imports.add(sig)
        clean_body.append(stmt)
      else:
        clean_body.append(stmt)

    return updated_node.with_changes(body=clean_body)

  def _append_injection(self, injections_list: List[cst.CSTNode], node: cst.CSTNode) -> None:
    injections_list.append(node)
