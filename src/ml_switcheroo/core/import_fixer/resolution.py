"""Import Resolution Logic.

This module centralizes the decision-making process for imports. It allows the current
codebase to be analyzed *before* transformation commits to determine imports.
"""

import libcst as cst
import pydantic

from ml_switcheroo.core.scanners import SimpleNameScanner, get_full_name
from ml_switcheroo.semantics.manager import SemanticsManager


class ImportReq(pydantic.BaseModel):
  """Represent a normalized import requirement.

  Can represent `import module as alias` or `from module import sub as alias`.
  """

  model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

  module: str
  subcomponent: str | None = None
  alias: str | None = None

  def __init__(
    self,
    module: str,
    subcomponent: str | None = None,
    alias: str | None = None,
  ) -> None:
    """Initialize ImportReq.

    Args:
        module: Root module name.
        subcomponent: Optional submodule or imported symbol name.
        alias: Optional local alias name.
    """
    super().__init__(module=module, subcomponent=subcomponent, alias=alias)

  @property
  def signature(self) -> str:
    """Unique signature for deduplication.

    Returns:
        str: Unique import signature string.
    """
    base = self.module
    if self.subcomponent:
      base += f".{self.subcomponent}"

    if self.alias:
      # Check for redundancy
      target = self.subcomponent if self.subcomponent else self.module.split(".")[-1]
      if self.alias != target:
        return f"{base} : {self.alias}"
    return base


class ResolutionPlan(pydantic.BaseModel):
  """Execute strategy for the ImportFixer to execute."""

  model_config = pydantic.ConfigDict(extra="forbid")

  required_imports: list[ImportReq] = pydantic.Field(default_factory=list)
  mappings: dict[str, ImportReq] = pydantic.Field(default_factory=dict)
  path_to_alias: dict[str, str] = pydantic.Field(default_factory=dict)

  def __init__(
    self,
    required_imports: list[ImportReq] | None = None,
    mappings: dict[str, ImportReq] | None = None,
    path_to_alias: dict[str, str] | None = None,
  ) -> None:
    """Initialize ResolutionPlan.

    Args:
        required_imports: List of required import specifications.
        mappings: Mapping of source import paths to target ImportReq objects.
        path_to_alias: Mapping of module paths to local alias names.
    """
    super().__init__(
      required_imports=required_imports if required_imports is not None else [],
      mappings=mappings if mappings is not None else {},
      path_to_alias=path_to_alias if path_to_alias is not None else {},
    )


class _QualNameScanner(cst.CSTVisitor):
  """Execute implementation detail."""

  def __init__(self, target_path: str):
    """Execute implementation detail.

    Args:
        target_path (str): The target attribute path to scan for.
    """
    self.target_path = target_path
    self.found = False

  def visit_Attribute(self, node: cst.Attribute) -> None:
    """Execute implementation detail.

    Args:
        node (cst.Attribute): The CST Attribute node.
    """
    if self.found:
      return
    try:
      name = get_full_name(node)
      if name == self.target_path or name.startswith(f"{self.target_path}."):
        self.found = True
    except (AttributeError, TypeError, ValueError):
      pass

  def visit_Name(self, node: cst.Name) -> None:
    """Execute implementation detail.

    Args:
        node (cst.Name): The CST Name node.
    """
    if self.found:
      return
    if node.value == self.target_path:
      self.found = True


class ImportResolver:
  """Execute implementation detail."""

  def __init__(self, semantics: SemanticsManager):
    """Execute implementation detail.

    Args:
        semantics (SemanticsManager): The semantics manager to use for resolution.
    """
    self.semantics = semantics

  def resolve(self, tree: cst.Module, target_framework: str) -> ResolutionPlan:
    """Execute implementation detail.

    Args:
        tree (cst.Module): The CST module to scan.
        target_framework (str): The target framework name.

    Returns:
        ResolutionPlan: The planned resolution for imports.
    """
    required: list[ImportReq] = []
    path_to_alias: dict[str, str] = {}
    mappings: dict[str, ImportReq] = {}

    # 1. Framework Base Check
    if self._is_used(tree, target_framework):
      required.append(ImportReq(module=target_framework))

    # 2. Framework Alias Check
    aliases = self.semantics.get_framework_aliases()
    if target_framework in aliases:
      mod_path, alias_name = aliases[target_framework]
      path_to_alias[mod_path] = alias_name

      if self._is_used(tree, alias_name) or self._is_path_used(tree, mod_path):
        # Resolving alias: e.g. jax.numpy -> jnp
        # if mod_path has dots, creating an ImportReq directly might create 'import jax.numpy as jnp'
        # which is valid but sometimes 'from jax import numpy as jnp' is preferred style.
        # Currently we stick to import-as.
        required.append(ImportReq(module=mod_path, alias=alias_name))

    # 3. Import Map (Submodules)
    raw_map = self.semantics.get_import_map(target_framework)

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
    """Execute implementation detail.

    Args:
        tree (cst.Module): The CST module to scan.
        name (str): The name to search for.

    Returns:
        bool: True if the name is used in the module.
    """
    scanner = SimpleNameScanner(name)
    tree.visit(scanner)
    return scanner.found

  def _is_path_used(self, tree: cst.Module, path: str) -> bool:
    """Execute implementation detail.

    Args:
        tree (cst.Module): The CST module to scan.
        path (str): The path to search for.

    Returns:
        bool: True if the path is used in the module.
    """
    scanner = _QualNameScanner(path)
    tree.visit(scanner)
    return scanner.found


def _deduplicate(reqs: list[ImportReq]) -> list[ImportReq]:
  """Execute implementation detail.

  Args:
      reqs (List[ImportReq]): The list of import requirements to deduplicate.

  Returns:
      List[ImportReq]: The deduplicated list of import requirements.
  """
  seen = set()
  out = []
  for r in reqs:
    sig = r.signature
    if sig not in seen:
      out.append(r)
      seen.add(sig)
  return out
