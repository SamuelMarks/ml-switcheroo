"""
Import Injection Mixin.

Handles the post-processing of the Module AST to:
1.  Verify usage of the target framework.
2.  Inject necessary top-level imports.
3.  Inject alias definitions.
4.  Perform final deduplication of import statements.
"""

from typing import List, Set

import libcst as cst

from ml_switcheroo.core.import_fixer.utils import (
  create_dotted_name,
  get_signature,
  is_docstring,
  is_future_import,
)
from ml_switcheroo.core.scanners import SimpleNameScanner


class InjectionMixin(cst.CSTTransformer):
  """
  Mixin for injecting imports at the Module level.
  """

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Post-process module to inject imports intelligently and perform final deduplication.

    Args:
        original_node: Original module logic.
        updated_node: Module after children processing.

    Returns:
        Modified module with injected imports.
    """
    injections: List[cst.CSTNode] = []

    # -- Step 1: Check Usage of Framework Base --
    if not self._target_found:
      scanner = SimpleNameScanner(self.target_fw)
      updated_node.visit(scanner)
      if scanner.found:
        node = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(self.target_fw))])])
        self._append_injection(injections, node)
        self._defined_names.add(self.target_fw)

    # -- Step 2: Check Usage of Framework Standard Aliases --
    if self.target_fw in self.alias_map:
      module_path, alias_name = self.alias_map[self.target_fw]

      scanner = SimpleNameScanner(alias_name)
      updated_node.visit(scanner)

      if scanner.found and alias_name not in self._defined_names:
        is_sub_match = False
        injection_node = None

        if "." in module_path:
          parts = module_path.rsplit(".", 1)
          if len(parts) == 2 and parts[1] == alias_name:
            injection_node = cst.SimpleStatementLine(
              body=[
                cst.ImportFrom(
                  module=create_dotted_name(parts[0]),
                  names=[cst.ImportAlias(name=cst.Name(alias_name))],
                )
              ]
            )
            is_sub_match = True

        if not is_sub_match:
          asname_node = None
          if alias_name != module_path:
            asname_node = cst.AsName(name=cst.Name(alias_name))

          injection_node = cst.SimpleStatementLine(
            body=[
              cst.Import(
                names=[
                  cst.ImportAlias(
                    name=create_dotted_name(module_path),
                    asname=asname_node,
                  )
                ]
              )
            ]
          )

        if injection_node:
          self._append_injection(injections, injection_node)
          self._defined_names.add(alias_name)

    # -- Step 3: Data-Driven Submodule Injection from Tier Resolution --
    for _, defn in self.submodule_map.items():
      tgt_root, tgt_sub, default_alias = defn

      if default_alias:
        desired_name = default_alias
        if desired_name in self._defined_names:
          continue

        scanner = SimpleNameScanner(desired_name)
        updated_node.visit(scanner)

        if scanner.found:
          injection_node = None
          if tgt_sub:
            asname_node = None
            if default_alias != tgt_sub:
              asname_node = cst.AsName(name=cst.Name(default_alias))

            injection_node = cst.SimpleStatementLine(
              body=[
                cst.ImportFrom(
                  module=create_dotted_name(tgt_root),
                  names=[cst.ImportAlias(name=cst.Name(tgt_sub), asname=asname_node)],
                )
              ]
            )
          else:
            asname_node = None
            if default_alias != tgt_root.split(".")[0]:
              asname_node = cst.AsName(name=cst.Name(default_alias))

            injection_node = cst.SimpleStatementLine(
              body=[
                cst.Import(
                  names=[
                    cst.ImportAlias(
                      name=create_dotted_name(tgt_root),
                      asname=asname_node,
                    )
                  ]
                )
              ]
            )

          if injection_node:
            self._append_injection(injections, injection_node)
            self._defined_names.add(desired_name)

    # Determine insertion point
    body_stats = list(updated_node.body)
    insert_idx = 0

    for i, stmt in enumerate(body_stats):
      if is_docstring(stmt, i):
        insert_idx = i + 1
        continue
      if is_future_import(stmt):
        insert_idx = i + 1
        continue
      break

    merged_body = body_stats[:insert_idx] + injections + body_stats[insert_idx:]

    # -- Step 4: Final Deduplication Pass --
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
    """Helper to add injection safely if not duplicate."""
    src = get_signature(node)
    if src not in self._injected_code_sigs:
      injections_list.append(node)
      self._injected_code_sigs.add(src)
