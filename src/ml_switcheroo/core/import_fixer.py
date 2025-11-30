"""
Import Management Transformer for ml-switcheroo.

This module is responsible for reading the AST's import references and remapping them
based on the data provided by SemanticsManager (Feature 024).

It performs:
1.  **Pruning**: Removes source framework imports (e.g. `import torch`) unless preserved.
2.  **Remapping**: Maps submodules (e.g. `from torch import nn` -> `from flax import linen as nn`).
3.  **Smart Injection**: Dynamically injects target framework imports and standard aliases
    (e.g., `import tensorflow as tf`, `import mlx.core as mx`) only when used in the code body.

Dependencies:
    Uses `ml_switcheroo.core.scanners` to detect name usage in the code body.
"""

from typing import Union, Optional, Tuple, Dict, Set, List

import libcst as cst
from ml_switcheroo.core.scanners import SimpleNameScanner, get_full_name

# Standard conventions for framework aliases
# key: target_framework_name
# value: (module_to_import, alias_name)
FRAMEWORK_ALIAS_MAP = {
  "jax": ("jax.numpy", "jnp"),
  "tensorflow": ("tensorflow", "tf"),
  "mlx": ("mlx.core", "mx"),
  "numpy": ("numpy", "np"),
}


class ImportFixer(cst.CSTTransformer):
  """
  LibCST Transformer that manages top-level imports.

  Attributes:
      source_fw (str): The root package name to remove (e.g., "torch").
      target_fw (str): The root package name to ensure exists (e.g., "jax").
      submodule_map (Dict): Map of "source.mod" -> (target_root, target_sub, alias).
      preserve_source (bool): If True, prevents automatic removal of source_fw imports.
      _target_found (bool): Result tracker for idempotency.
      _defined_names (Set[str]): Tracks names explicitly defined by imports to prevent re-injection.
  """

  def __init__(
    self,
    source_fw: str,
    target_fw: str,
    submodule_map: Dict[str, Tuple[str, Optional[str], Optional[str]]],
    preserve_source: bool = False,
  ):
    """
    Initializes the ImportFixer.

    Args:
        source_fw: The framework string to strip.
        target_fw: The framework string to inject.
        submodule_map: Data-driven remapping dictionary from SemanticsManager.
        preserve_source: Whether to keep source imports even if matched.
    """
    self.source_fw = source_fw
    self.target_fw = target_fw
    self.submodule_map = submodule_map
    self.preserve_source = preserve_source
    self._target_found = False
    self._defined_names: Set[str] = set()

  def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> Union[cst.Import, cst.RemovalSentinel]:
    """
    Inspects 'import ...' statements.
    Checks aliases against submodule_map or prunes matches to source_fw.

    Args:
        original_node: The original CST node.
        updated_node: The CST node with transformed children.

    Returns:
        The modified node or RemovalSentinel if prune is required.
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
        if not new_asname and default_alias:
          new_asname = cst.AsName(name=cst.Name(default_alias))

        new_alias = cst.ImportAlias(name=self._create_dotted_name(new_name_str), asname=new_asname)
        self._track_definition(new_alias)  # Track the new name
        new_aliases.append(new_alias)
        continue

      # 3. Prune Source Root (e.g. import torch)
      if root_pkg == self.source_fw:
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
    Inspects 'from ... import ...' statements.
    Checks module+name against submodule_map.

    Args:
        original_node: The original CST node.
        updated_node: The CST node with transformed children.

    Returns:
        The modified node or RemovalSentinel.
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
    if root_pkg == self.source_fw:
      if self.preserve_source:
        return updated_node
      return cst.RemoveFromParent()

    if root_pkg == self.target_fw:
      self._target_found = True

    return updated_node

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Post-process module to inject imports intelligently.

    Logic:
    1. Identifies insertion point (after __future__ and docstrings).
    2. Injects 'import target' only if needed (not found AND used).
    3. Injects aliases (e.g. 'jnp', 'tf', 'mx') if detected in usage.
    4. Injects submodule mappings (e.g., from flax import linen) if usage requires it.

    Args:
        original_node: The original CST module.
        updated_node: The CST module after children transformations.

    Returns:
        The final CST module with injected imports.
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

    # -- Step 2: Check Usage of Framework Standard Aliases (e.g. 'jnp', 'tf', 'mx') --
    if self.target_fw in FRAMEWORK_ALIAS_MAP:
      module_path, alias_name = FRAMEWORK_ALIAS_MAP[self.target_fw]

      scanner = SimpleNameScanner(alias_name)
      updated_node.visit(scanner)

      if scanner.found and alias_name not in self._defined_names:
        injections.append(
          cst.SimpleStatementLine(
            body=[
              cst.Import(
                names=[
                  cst.ImportAlias(
                    name=self._create_dotted_name(module_path),
                    asname=cst.AsName(name=cst.Name(alias_name)),
                  )
                ]
              )
            ]
          )
        )

    # -- Step 3: Data-Driven Submodule Injection (Smart 'From' Logic) --
    # Iterate over submodule_map to see if any mapped aliases are used but undefined
    # Map Format: "source.mod" -> (target_root, target_sub, alias)
    for _, defn in self.submodule_map.items():
      tgt_root, tgt_sub, default_alias = defn

      # We care about the name used in code. Usually 'default_alias'.
      if default_alias:
        desired_name = default_alias
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
            # import {tgt_root}
            # import {tgt_root} as {alias}
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
