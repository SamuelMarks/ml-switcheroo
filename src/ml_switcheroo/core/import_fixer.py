"""
Import Management Transformer for ml-switcheroo.

This module is responsible for reading the AST's import references and remapping them
based on the data provided by SemanticsManager (Feature 024).
It also handles collapsing fully qualified paths (e.g. `torch.nn.Linear`) into
aliases (e.g. `nn.Linear`) based on content injection logic.
"""

from typing import Union, Optional, Tuple, Dict, Set, List

import libcst as cst
from ml_switcheroo.core.scanners import SimpleNameScanner, get_full_name


class ImportFixer(cst.CSTTransformer):
  """
  LibCST Transformer that manages top-level imports and path shortening.
  """

  def __init__(
    self,
    source_fws: Union[str, List[str]],
    target_fw: str,
    submodule_map: Dict[str, Tuple[str, Optional[str], Optional[str]]],
    alias_map: Optional[Dict[str, Tuple[str, str]]] = None,
    preserve_source: bool = False,
  ):
    """
    Initializes the ImportFixer.

    Args:
        source_fws: Single string or List of framework strings to strip (e.g. ['flax', 'jax']).
        target_fw: The framework string to inject.
        submodule_map: Data-driven remapping dictionary from SemanticsManager.
        alias_map: Configuration for standard framework aliases (e.g. jax->(jax.numpy, jnp)).
                   Added in Feature 07 to replace hardcoded map.
        preserve_source: Whether to keep source imports even if matched.
    """
    if isinstance(source_fws, str):
      self.source_fws = {source_fws}
    else:
      self.source_fws = set(source_fws)

    self.target_fw = target_fw
    # Map of "fully.qualified.path" -> ("root", "sub", "alias")
    self.submodule_map = submodule_map
    self.alias_map = alias_map or {}
    self.preserve_source = preserve_source
    self._target_found = False
    self._defined_names: Set[str] = set()

    # Build Collapsing Map: target_full_path -> alias
    self._path_to_alias = {}

    # 1. From Submodules Config
    for _, (root, sub, alias) in self.submodule_map.items():
      if alias:
        full_path = f"{root}.{sub}" if sub else root
        self._path_to_alias[full_path] = alias

    # 2. From Framework Aliases Config
    for _, (mod, alias) in self.alias_map.items():
      if alias:
        self._path_to_alias[mod] = alias

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """
    Collapses fully qualified paths to aliases if configured.
    e.g. `torch.nn.Linear` -> `nn.Linear` if `torch.nn` -> `nn` alias exists for this target.
    """
    full_name = get_full_name(updated_node)
    if not full_name:
      return updated_node

    parts = full_name.split(".")
    # Try matching prefixes from longest to shortest
    for i in range(len(parts) - 1, 0, -1):
      prefix = ".".join(parts[:i])

      if prefix in self._path_to_alias:
        alias = self._path_to_alias[prefix]
        suffix_parts = parts[i:]

        # Construct collapsed node: alias.Remainder
        new_node = cst.Name(alias)
        for part in suffix_parts:
          new_node = cst.Attribute(value=new_node, attr=cst.Name(part))
        return new_node

    return updated_node

  def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> Union[cst.Import, cst.RemovalSentinel]:
    """
    Inspects ``import ...`` statements.
    Checks aliases against submodule_map or prunes matches to source_fw.
    """
    new_aliases = []

    for alias in updated_node.names:
      full_name = get_full_name(alias.name)
      root_pkg = full_name.split(".")[0]

      # Track potential definitions
      self._track_definition(alias)

      # 1. Check if Target is already imported
      if root_pkg == self.target_fw:
        self._target_found = True
        new_aliases.append(alias)
        continue

      # 2. Check for Specific Mapping (e.g. import torch.nn)
      mapping = self.submodule_map.get(full_name)
      if mapping:
        tgt_root, tgt_sub, default_alias = mapping

        # Construct flattened target name: flax.linen
        new_name_str = f"{tgt_root}.{tgt_sub}" if tgt_sub else tgt_root

        # Determine alias
        new_asname = alias.asname
        # Use default alias if present and no user alias
        if not new_asname and default_alias:
          new_asname = cst.AsName(name=cst.Name(default_alias))

        new_alias = cst.ImportAlias(name=self._create_dotted_name(new_name_str), asname=new_asname)
        self._track_definition(new_alias)  # Track the new name
        new_aliases.append(new_alias)
        continue

      # 3. Prune Source Roots
      if root_pkg in self.source_fws:
        if self.preserve_source:
          new_aliases.append(alias)
        continue

      # Keep unrelated imports
      new_aliases.append(alias)

    if not new_aliases:
      return cst.RemoveFromParent()

    return updated_node.with_changes(names=new_aliases)

  def leave_ImportFrom(
    self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
  ) -> Union[cst.ImportFrom, cst.RemovalSentinel]:
    """
    Inspects ``from ... import ...`` statements.
    Checks module+name against submodule_map.
    """
    if not updated_node.module:
      return updated_node

    module_name = get_full_name(updated_node.module)
    root_pkg = module_name.split(".")[0]

    # Track existing defs
    for alias in updated_node.names:
      if isinstance(alias, cst.ImportAlias):
        self._track_definition(alias)

    # Check explicit submodule import (from torch import nn)
    if len(updated_node.names) == 1 and isinstance(updated_node.names[0], cst.ImportAlias):
      import_name = updated_node.names[0].name.value
      lookup_key = f"{module_name}.{import_name}"

      mapping = self.submodule_map.get(lookup_key)
      if mapping:
        tgt_root, tgt_sub, default_alias = mapping

        # e.g. from flax import linen as nn
        existing_alias = updated_node.names[0].asname
        final_alias = existing_alias

        if not final_alias and default_alias:
          final_alias = cst.AsName(name=cst.Name(default_alias))

        new_node = cst.ImportFrom(
          module=cst.Name(tgt_root),
          names=[
            cst.ImportAlias(
              name=cst.Name(tgt_sub) if tgt_sub else cst.Name(tgt_root),
              asname=final_alias,
            )
          ],
        )
        # Track the NEW definition so we don't duplicate inject it later
        self._track_definition(new_node.names[0])
        return new_node

    # Prune source framework imports
    if root_pkg in self.source_fws:
      if self.preserve_source:
        return updated_node
      return cst.RemoveFromParent()

    if root_pkg == self.target_fw:
      self._target_found = True

    return updated_node

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Post-process module to inject imports intelligently.
    """
    injections: List[cst.CSTNode] = []

    # -- Step 1: Check Usage of Framework Base (e.g. 'jax') --
    if not self._target_found:
      scanner = SimpleNameScanner(self.target_fw)
      updated_node.visit(scanner)
      if scanner.found:
        injections.append(
          cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(self.target_fw))])])
        )
        # Prevent re-injection by Step 2 if alias matches root
        self._defined_names.add(self.target_fw)

    # -- Step 2: Check Usage of Framework Standard Aliases (e.g. 'jnp', 'tf', 'mx') --
    # Logic driven by self.alias_map loaded from SemanticsManager (default or custom)
    if self.target_fw in self.alias_map:
      module_path, alias_name = self.alias_map[self.target_fw]

      scanner = SimpleNameScanner(alias_name)
      updated_node.visit(scanner)

      if scanner.found and alias_name not in self._defined_names:
        # FIX: Avoid 'import torch as torch'.
        asname_node = None
        if alias_name != module_path:
          asname_node = cst.AsName(name=cst.Name(alias_name))

        injections.append(
          cst.SimpleStatementLine(
            body=[
              cst.Import(
                names=[
                  cst.ImportAlias(
                    name=self._create_dotted_name(module_path),
                    asname=asname_node,
                  )
                ]
              )
            ]
          )
        )
        self._defined_names.add(alias_name)

    # -- Step 3: Data-Driven Submodule Injection (Smart 'From' Logic) --
    # Iterate over submodule_map to see if any mapped aliases are used but undefined
    # Map Format: "source.mod" -> (target_root, target_sub, alias)
    for _, defn in self.submodule_map.items():
      tgt_root, tgt_sub, default_alias = defn

      if default_alias:
        desired_name = default_alias
        # Check usage
        scanner = SimpleNameScanner(desired_name)
        updated_node.visit(scanner)

        if scanner.found and desired_name not in self._defined_names:
          # Decide between `import` vs `from ... import` based on sub path existence
          if tgt_sub:
            # Case A: Submodule Import
            # from {tgt_root} import {tgt_sub}
            # from {tgt_root} import {tgt_sub} as {alias}
            asname_node = None
            if default_alias != tgt_sub:
              asname_node = cst.AsName(name=cst.Name(default_alias))

            injection_node = cst.SimpleStatementLine(
              body=[
                cst.ImportFrom(
                  module=cst.Name(tgt_root),
                  names=[cst.ImportAlias(name=cst.Name(tgt_sub), asname=asname_node)],
                )
              ]
            )
          else:
            # Case B: Root Import
            asname_node = None
            if default_alias != tgt_root:
              asname_node = cst.AsName(name=cst.Name(default_alias))

            injection_node = cst.SimpleStatementLine(
              body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(tgt_root), asname=asname_node)])]
            )

          injections.append(injection_node)
          # Mark as defined so we don't inject twice if loops overlap
          self._defined_names.add(desired_name)

    if not injections:
      return updated_node

    # -- Step 4: Insert at correct location --
    insert_idx = 0
    body_stats = list(updated_node.body)

    for i, stmt in enumerate(body_stats):
      # Skip Module Docstring (SimpleStmt with Expr with String)
      if self._is_docstring(stmt, i):
        insert_idx = i + 1
        continue

      # Skip __future__ imports (must be first code)
      if self._is_future_import(stmt):
        insert_idx = i + 1
        continue

      # Stop at first real code or other imports
      break

    # Assemble new body
    new_body = body_stats[:insert_idx] + injections + body_stats[insert_idx:]

    return updated_node.with_changes(body=new_body)

  def _track_definition(self, alias_node: cst.ImportAlias) -> None:
    """Records a name being defined by an import statement."""
    if alias_node.asname:
      self._defined_names.add(alias_node.asname.name.value)
    else:
      # "import a.b" defines "a" in local scope
      # "from a import b" defines "b"
      name_val = get_full_name(alias_node.name)
      # If it's a dotted import (import a.b), only root is bound
      self._defined_names.add(name_val.split(".")[0])

  def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:
    """Creates a CST node for a dotted path."""
    parts = name_str.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _is_docstring(self, node: cst.CSTNode, idx: int) -> bool:
    """
    Checks if the statement is a module level docstring.
    Must be the first non-comment statement in file (index 0).
    """
    if idx != 0:
      return False
    if isinstance(node, cst.SimpleStatementLine):
      if len(node.body) == 1 and isinstance(node.body[0], cst.Expr):
        expr = node.body[0].value
        # Check for string literal (triple-quoted or single)
        if isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
          return True
    return False

  def _is_future_import(self, node: cst.CSTNode) -> bool:
    """Checks if the statement is 'from __future__ import ...'."""
    if isinstance(node, cst.SimpleStatementLine):
      for small_stmt in node.body:
        if isinstance(small_stmt, cst.ImportFrom):
          if small_stmt.module and small_stmt.module.value == "__future__":
            return True
    return False
