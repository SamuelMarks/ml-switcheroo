"""
Import Resolution Logic.

This module centralizes the decision-making process for imports. It allows the current
codebase to be analyzed *before* transformation commits to determine imports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import libcst as cst

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.scanners import SimpleNameScanner, get_full_name


@dataclass(frozen=True)
class ImportReq:
  """
  Represents a normalized import requirement.
  Can represent `import module as alias` or `from module import sub as alias`.
  """

  module: str
  subcomponent: Optional[str] = None
  alias: Optional[str] = None

  @property
  def signature(self) -> str:
    """Unique signature for deduplication."""
    base = self.module
    if self.subcomponent:
      base += f".{self.subcomponent}"

    if self.alias:
      # Check for redundancy
      target = self.subcomponent if self.subcomponent else self.module.split(".")[-1]
      if self.alias != target:
        return f"{base} : {self.alias}"
    return base


@dataclass
class ResolutionPlan:
  """The strategy for the ImportFixer to execute."""

  required_imports: List[ImportReq] = field(default_factory=list)
  mappings: Dict[str, ImportReq] = field(default_factory=dict)
  path_to_alias: Dict[str, str] = field(default_factory=dict)


class _QualNameScanner(cst.CSTVisitor):
  def __init__(self, target_path: str):
    self.target_path = target_path
    self.found = False

  def visit_Attribute(self, node: cst.Attribute) -> None:
    if self.found:
      return
    try:
      name = get_full_name(node)
      if name == self.target_path or name.startswith(f"{self.target_path}."):
        self.found = True
    except:
      pass

  def visit_Name(self, node: cst.Name) -> None:
    if self.found:
      return
    if node.value == self.target_path:
      self.found = True


class ImportResolver:
  def __init__(self, semantics: SemanticsManager):
    self.semantics = semantics

  def resolve(self, tree: cst.Module, target_fw: str) -> ResolutionPlan:
    required: List[ImportReq] = []
    path_to_alias: Dict[str, str] = {}
    mappings: Dict[str, ImportReq] = {}

    # 1. Framework Base Check
    if self._is_used(tree, target_fw):
      required.append(ImportReq(module=target_fw))

    # 2. Framework Alias Check
    aliases = self.semantics.get_framework_aliases()
    if target_fw in aliases:
      mod_path, alias_name = aliases[target_fw]
      path_to_alias[mod_path] = alias_name

      if self._is_used(tree, alias_name) or self._is_path_used(tree, mod_path):
        # Resolving alias: e.g. jax.numpy -> jnp
        # if mod_path has dots, creating an ImportReq directly might create 'import jax.numpy as jnp'
        # which is valid but sometimes 'from jax import numpy as jnp' is preferred style.
        # Currently we stick to import-as.
        required.append(ImportReq(module=mod_path, alias=alias_name))

    # 3. Import Map (Submodules)
    raw_map = self.semantics.get_import_map(target_fw)

    for src_path, (tgt_root, tgt_sub, tgt_alias) in raw_map.items():
      req = ImportReq(module=tgt_root, subcomponent=tgt_sub, alias=tgt_alias)
      mappings[src_path] = req

      full_target_path = f"{tgt_root}.{tgt_sub}" if tgt_sub else tgt_root

      # Injection Logic
      # We inject if:
      # 1. The code uses the alias name explicitly (e.g. 'jnp').
      # 2. OR the code uses the full path (e.g. 'jax.numpy'), which AttributeMixin will collapse to alias.
      check_name = tgt_alias
      if not check_name:
        check_name = tgt_sub if tgt_sub else tgt_root.split(".")[-1]

      if self._is_used(tree, check_name) or (tgt_alias and self._is_path_used(tree, full_target_path)):
        # Prefer 'from X import Y' style for submodules in injection if sub exists
        # This aligns with test expectations like 'from flax import nnx'
        required.append(req)

      if tgt_alias:
        path_to_alias[full_target_path] = tgt_alias

    return ResolutionPlan(required_imports=_deduplicate(required), mappings=mappings, path_to_alias=path_to_alias)

  def _is_used(self, tree: cst.Module, name: str) -> bool:
    scanner = SimpleNameScanner(name)
    tree.visit(scanner)
    return scanner.found

  def _is_path_used(self, tree: cst.Module, path: str) -> bool:
    scanner = _QualNameScanner(path)
    tree.visit(scanner)
    return scanner.found


def _deduplicate(reqs: List[ImportReq]) -> List[ImportReq]:
  seen = set()
  out = []
  for r in reqs:
    sig = r.signature
    if sig not in seen:
      out.append(r)
      seen.add(sig)
  return out
