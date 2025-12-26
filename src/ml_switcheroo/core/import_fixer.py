"""
Import Management Transformer for ml-switcheroo.

This module is responsible for reading the AST's import references and remapping them
based on the data provided by SemanticsManager.
It also handles collapsing fully qualified paths (e.g. `torch.nn.Linear`) into
aliases (e.g. `nn.Linear`) based on content injection logic.

Updates:
- Integrates with Tier-Based Subscription output from Manager.
- Maintains support for intelligent alias collapsing and smart injection.
- Prevents duplicate insertions via node similarity checking.
- Fix: Registers transformed imports in `_injected_code_sigs` to prevent re-injection.
- Fix: Unwraps SimpleStatementLine during signature generation for robust deduplication.
- Fix: Normalizes whitespace in signatures to ensure consistent duplicate detection.
- Fix: Robustly handles `AssignTarget` in `_track_definition` to correctly detect defined aliases.
- Refactor: Implements a final top-level deduplication pass in `leave_Module` to strictly prevent duplicates.
- Fix: Corrected redundancy check logic for root imports (e.g. `import torch.nn as nn` is NOT redundant).
"""

from typing import Union, Optional, Tuple, Dict, Set, List
import re

import libcst as cst
from ml_switcheroo.core.scanners import SimpleNameScanner, get_full_name
from ml_switcheroo.utils.node_diff import capture_node_source


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
                       Format: { "source.path": ("target_root", "target_sub", "alias") }
        alias_map: Configuration for standard framework aliases (e.g. jax->(jax.numpy, jnp)).
        preserve_source: Whether to keep source imports even if matched.
    """
    if isinstance(source_fws, str):
      self.source_fws = {source_fws}
    else:
      self.source_fws = set(source_fws)

    self.target_fw = target_fw
    # IMPORTANT: Filter submodule_map to only include entries RELEVANT to the target_fw.
    # SemanticsManager.get_import_map does filtering based on provider, but the fixer logic
    # here assumes ANY match in submodule_map is a replacement instruction.
    # If using a mock map in tests, this filtering is redundant but safe.
    self.submodule_map = submodule_map
    self.alias_map = alias_map or {}
    self.preserve_source = preserve_source
    self._target_found = False
    self._defined_names: Set[str] = set()
    # Use string tracking to prevent dupes in complex scenarios not covered by _defined_names
    self._injected_code_sigs: Set[str] = set()

    # Build Collapsing Map: target_full_path -> alias
    self._path_to_alias = {}

    # 1. From Submodules Config (Result of Tier Mapping)
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
    # Use original_node to get the full untransformed path (e.g. 'torch.nn.functional.relu')
    full_name = get_full_name(original_node)
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

    # Flag to track if we injected a strict target replacement
    # If so, we should NOT preserve the source even if preserve_source is True
    # because it has been actively replaced by a target equivalent.
    replacement_occurred = False

    for alias in updated_node.names:
      full_name = get_full_name(alias.name)
      root_pkg = full_name.split(".")[0]

      # Track potential definitions to prevent re-injection later
      self._track_definition(alias)

      # 1. Check if Target is already imported
      if root_pkg == self.target_fw:
        self._target_found = True
        new_aliases.append(alias)
        continue

      # 2. Check for Specific Mapping (e.g. import torch.nn)
      # This logic leverages the Tier Mapping provided by Manager
      mapping = self.submodule_map.get(full_name)
      if mapping:
        tgt_root, tgt_sub, default_alias = mapping

        # Construct flattened target name: flax.linen
        new_name_str = f"{tgt_root}.{tgt_sub}" if tgt_sub else tgt_root

        # Determine alias
        new_asname = None
        if default_alias:
          # Determine if alias is redundant
          # Case: Import statement (tgt_sub is None, e.g. import torch.nn)
          # Python behavior: `import A.B` binds `A` locally.
          # If we want `B`, we must use `as B` or `as alias`.
          # It is only redundant if `alias` matches the root package.
          is_redundant = False
          root_package = new_name_str.split(".")[0]

          if default_alias == root_package:
            is_redundant = True

          if not is_redundant:
            new_asname = cst.AsName(name=cst.Name(default_alias))

        elif alias.asname:
          # Fallback: keep user alias if no imperative alias defined
          new_asname = alias.asname

        new_alias = cst.ImportAlias(name=self._create_dotted_name(new_name_str), asname=new_asname)
        self._track_definition(new_alias)  # Track the new name
        new_aliases.append(new_alias)

        replacement_occurred = True
        continue

      # 3. Prune Source Roots
      if root_pkg in self.source_fws:
        # Only preserve if we haven't already replaced it with a target equivalent
        # AND preservation is requested
        if self.preserve_source and not replacement_occurred:
          new_aliases.append(alias)
        continue

      # Keep unrelated imports
      new_aliases.append(alias)

    if not new_aliases:
      return cst.RemoveFromParent()

    res_node = updated_node.with_changes(names=new_aliases)
    # Register the transformed node to prevent re-injection of the same signature
    self._injected_code_sigs.add(self._get_signature(res_node))
    return res_node

  def leave_ImportFrom(
    self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
  ) -> Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel]:
    """
    Inspects ``from ... import ...`` statements.
    Checks module+name against submodule_map.
    """
    if not updated_node.module:
      return updated_node

    module_name = get_full_name(updated_node.module)
    root_pkg = module_name.split(".")[0]

    # Capture current source representation to prevent re-injection
    self._injected_code_sigs.add(self._get_signature(updated_node))

    # Track existing defs
    for alias in updated_node.names:
      if isinstance(alias, cst.ImportAlias):
        self._track_definition(alias)

    # Check explicit submodule import (from torch import nn)
    if len(updated_node.names) == 1 and isinstance(updated_node.names[0], cst.ImportAlias):
      import_name = updated_node.names[0].name.value
      # If using 'from X import Y', full name is X.Y
      lookup_key = f"{module_name}.{import_name}"

      mapping = self.submodule_map.get(lookup_key)
      if mapping:
        tgt_root, tgt_sub, default_alias = mapping

        final_alias = None

        # Check redundancy logic
        if default_alias:
          is_redundant = False

          if tgt_sub:
            # Case: from A import B (as B)
            if default_alias == tgt_sub:
              is_redundant = True
          elif tgt_sub is None:
            # Case: import A (as A) - e.g. converting 'from flax import nnx' -> 'import torch.nn'
            # 'import torch.nn' binds 'torch'. We need alias 'nn' to access 'torch.nn' easily.
            # Alias is only redundant if it is equal to the root package name.
            root_package = tgt_root.split(".")[0]
            if default_alias == root_package:
              is_redundant = True

          if not is_redundant:
            final_alias = cst.AsName(name=cst.Name(default_alias))
        elif updated_node.names[0].asname:
          final_alias = updated_node.names[0].asname

        # Logic: Handle case where target is a direct import (tgt_sub is None)
        # Example: Mapped 'flax.nnx' -> 'torch.nn' (root='torch.nn', sub=None)
        # Source: 'from flax import nnx' -> Target: 'import torch.nn as nn'
        if tgt_sub is None:
          # Converting FROM import to IMPORT.
          new_node = cst.Import(names=[cst.ImportAlias(name=self._create_dotted_name(tgt_root), asname=final_alias)])
          # Track definition
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])

            # Register the source of this new node to prevent duplicate injection
          self._injected_code_sigs.add(self._get_signature(new_node))
          return new_node

        else:
          # Standard 'from ... import ...' replacement
          new_node = cst.ImportFrom(
            module=self._create_dotted_name(tgt_root),
            names=[
              cst.ImportAlias(
                name=cst.Name(tgt_sub),
                asname=final_alias,
              )
            ],
          )
          # Track definition
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])
            # Register the source of this new node
          self._injected_code_sigs.add(self._get_signature(new_node))
          return new_node

    # Prune source framework imports
    if root_pkg in self.source_fws:
      # Since we handle replacements above, if we reach here it means no mapping triggered replacement.
      # So we rely on preserve_source flag.
      if self.preserve_source:
        return updated_node
      return cst.RemoveFromParent()

    if root_pkg == self.target_fw:
      self._target_found = True

    return updated_node

  def _get_signature(self, node: cst.CSTNode) -> str:
    """
    Computes a deduplication signature for an Import/ImportFrom node.

    Handles unwrapping SimpleStatementLine containers and normalizes whitespace
    to ensure robust duplicate detection (e.g. "import a as b" vs "import a as b\n").
    """
    target = node
    # Unwrap Wrapper if present, recursively if needed (though usually 1 level)
    while isinstance(target, cst.SimpleStatementLine) and len(target.body) > 0:
      target = target.body[0]

    src = capture_node_source(target)
    # Removing whitespace and newlines for normalization
    return " ".join(src.split())

  def _append_injection(self, injections_list, node):
    """Helper to add injection safely if not duplicate."""
    src = self._get_signature(node)

    if src not in self._injected_code_sigs:
      injections_list.append(node)
      self._injected_code_sigs.add(src)

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Post-process module to inject imports intelligently and perform final deduplication.
    """
    injections: List[cst.CSTNode] = []

    # -- Step 1: Check Usage of Framework Base (e.g. 'jax') --
    if not self._target_found:
      scanner = SimpleNameScanner(self.target_fw)
      updated_node.visit(scanner)
      if scanner.found:
        node = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(self.target_fw))])])
        self._append_injection(injections, node)
        self._defined_names.add(self.target_fw)

        # -- Step 2: Check Usage of Framework Standard Aliases (e.g. 'jnp', 'tf') --
    if self.target_fw in self.alias_map:
      module_path, alias_name = self.alias_map[self.target_fw]

      scanner = SimpleNameScanner(alias_name)
      updated_node.visit(scanner)

      # FIX: Check if alias_name is already defined in _defined_names to prevent double injection
      # Logic: If 'import flax.nnx as nn' happened (defining 'nn')
      # and alias_map says jax->nn (unlikely but possible clash), skip injection.
      if scanner.found and alias_name not in self._defined_names:
        # Logic: If alias matches a submodule, prefer 'from X import Y'
        is_sub_match = False
        injection_node = None

        if "." in module_path:
          parts = module_path.rsplit(".", 1)
          if len(parts) == 2 and parts[1] == alias_name:
            injection_node = cst.SimpleStatementLine(
              body=[
                cst.ImportFrom(
                  module=self._create_dotted_name(parts[0]),
                  names=[cst.ImportAlias(name=cst.Name(alias_name))],
                )
              ]
            )
            is_sub_match = True

        if not is_sub_match:
          # Fallback to standard 'import X as Y'
          asname_node = None
          if alias_name != module_path:
            asname_node = cst.AsName(name=cst.Name(alias_name))

          injection_node = cst.SimpleStatementLine(
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

        if injection_node:
          self._append_injection(injections, injection_node)
          self._defined_names.add(alias_name)

    # -- Step 3: Data-Driven Submodule Injection from Tier Resolution --
    for _, defn in self.submodule_map.items():
      tgt_root, tgt_sub, default_alias = defn

      if default_alias:
        desired_name = default_alias
        # Check if already defined/imported to prevent double import
        if desired_name in self._defined_names:
          continue

        scanner = SimpleNameScanner(desired_name)
        updated_node.visit(scanner)

        if scanner.found:
          injection_node = None
          if tgt_sub:
            # Case A: Submodule Import
            asname_node = None
            if default_alias != tgt_sub:
              asname_node = cst.AsName(name=cst.Name(default_alias))

            injection_node = cst.SimpleStatementLine(
              body=[
                cst.ImportFrom(
                  module=self._create_dotted_name(tgt_root),
                  names=[cst.ImportAlias(name=cst.Name(tgt_sub), asname=asname_node)],
                )
              ]
            )
          else:
            # Case B: Root Import
            asname_node = None
            if default_alias != tgt_root.split(".")[0]:
              asname_node = cst.AsName(name=cst.Name(default_alias))

            injection_node = cst.SimpleStatementLine(
              body=[cst.Import(names=[cst.ImportAlias(name=self._create_dotted_name(tgt_root), asname=asname_node)])]
            )

          if injection_node:
            self._append_injection(injections, injection_node)
            self._defined_names.add(desired_name)

    # Determine insertion point (after docstring)
    body_stats = list(updated_node.body)
    insert_idx = 0

    for i, stmt in enumerate(body_stats):
      if self._is_docstring(stmt, i):
        insert_idx = i + 1
        continue
      if self._is_future_import(stmt):
        insert_idx = i + 1
        continue
      break

    # Construct the potential new body
    merged_body = body_stats[:insert_idx] + injections + body_stats[insert_idx:]

    # -- Step 4: Final Deduplication Pass --
    # Iterate through the top-level statements and enforce uniqueness on Imports.
    # This prevents the "double import" bug if both Transformation and Injection added the same line.

    clean_body = []
    seen_imports: Set[str] = set()

    for stmt in merged_body:
      is_import = False
      sig = ""

      # Check if statement is purely imports
      # We unwrap SimpleStatementLine to check children
      if isinstance(stmt, cst.SimpleStatementLine):
        all_imports = True
        for small in stmt.body:
          if not isinstance(small, (cst.Import, cst.ImportFrom)):
            all_imports = False
            break

        if all_imports and stmt.body:
          is_import = True
          sig = self._get_signature(stmt)

      if is_import:
        if sig in seen_imports:
          continue  # Skip duplicate
        seen_imports.add(sig)
        clean_body.append(stmt)
      else:
        clean_body.append(stmt)

    return updated_node.with_changes(body=clean_body)

  def _track_definition(self, alias_node: cst.ImportAlias) -> None:
    """Records a name being defined by an import statement."""
    if alias_node.asname:
      # Robust extraction: asname.name might be AssignTarget (wrapping Name) or Name directly
      target = alias_node.asname.name
      if isinstance(target, cst.AssignTarget):
        target = target.target

      if isinstance(target, cst.Name):
        self._defined_names.add(target.value)
    else:
      name_val = get_full_name(alias_node.name)
      # e.g. import torch.nn -> adds torch
      self._defined_names.add(name_val.split(".")[0])

  def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:
    """Creates a CST node for a dotted path."""
    parts = name_str.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _is_docstring(self, node: cst.CSTNode, idx: int) -> bool:
    if idx != 0:
      return False
    if isinstance(node, cst.SimpleStatementLine):
      if len(node.body) == 1 and isinstance(node.body[0], cst.Expr):
        expr = node.body[0].value
        if isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
          return True
    return False

  def _is_future_import(self, node: cst.CSTNode) -> bool:
    if isinstance(node, cst.SimpleStatementLine):
      for small_stmt in node.body:
        if isinstance(small_stmt, cst.ImportFrom):
          if small_stmt.module and small_stmt.module.value == "__future__":
            return True
    return False
