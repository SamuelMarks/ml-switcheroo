"""Import Fixer Package.

This package provides ``ImportFixer``, a single-inheritance ``cst.CSTTransformer``
composing three explicit collaborators over a shared ``ImportFixerState``:
1. ``ImportStatementRewriter``: rewriting and pruning ``Import`` and ``ImportFrom`` statements.
2. ``ImportInjector``: injecting missing top-level target imports into ``Module`` bodies.
3. ``AttributeAliasSimplifier``: collapsing qualified names to aliases and stripping re-exports.
"""

import typing

import libcst as cst

from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeAliasSimplifier
from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.imports_mixin import ImportStatementRewriter
from ml_switcheroo.core.import_fixer.injection_mixin import ImportInjector
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ImportResolver, ResolutionPlan


class ImportFixer(cst.CSTTransformer):
  """LibCST transformer managing imports and qualified attribute names via composition.

  Attributes:
      state: Shared ``ImportFixerState`` tracking resolution plan, defined names, and injections.
      attribute_simplifier: Collaborator simplifying ``cst.Attribute`` chains and aliases.
      import_rewriter: Collaborator rewriting and pruning ``Import`` and ``ImportFrom`` nodes.
      import_injector: Collaborator injecting missing top-level imports at ``Module`` exit.
  """

  def __init__(
    self,
    plan: ResolutionPlan,
    source_frameworks: str | typing.Iterable[str] | None = None,
    used_names: set[str] | None = None,
  ) -> None:
    """Initialize ``ImportFixer`` and its composed collaborator components.

    Args:
        plan: Pre-calculated ``ResolutionPlan`` describing required imports and mappings.
        source_frameworks: Framework(s) to explicitly strip imports for (e.g. ``'torch'``).
        used_names: Set of names referenced in the code body for dead-code elimination.
    """
    self.state: ImportFixerState = ImportFixerState.create(
      plan=plan,
      source_frameworks=source_frameworks,
      used_names=used_names,
    )
    self.attribute_simplifier: AttributeAliasSimplifier = AttributeAliasSimplifier(self.state)
    self.import_rewriter: ImportStatementRewriter = ImportStatementRewriter(self.state)
    self.import_injector: ImportInjector = ImportInjector(self.state)

  @property
  def plan(self) -> ResolutionPlan:
    """Return the active ``ResolutionPlan``.

    Returns:
        Active ``ResolutionPlan`` stored in ``self.state``.
    """
    return self.state.plan

  @property
  def source_frameworks(self) -> set[str]:
    """Return the configured source framework root names.

    Returns:
        Set of source framework names in ``self.state``.
    """
    return self.state.source_frameworks

  @property
  def used_names(self) -> set[str]:
    """Return the set of body-referenced names used for dead-code elimination.

    Returns:
        Set of used identifier names in ``self.state``.
    """
    return self.state.used_names

  @property
  def target_framework(self) -> str | None:
    """Return the optional target framework identifier from ``self.state``.

    Returns:
        Target framework identifier or ``None``.
    """
    return self.state.target_framework

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """Delegate ``cst.Attribute`` alias collapsing and re-export cleanup to ``attribute_simplifier``.

    Args:
        original_node: Attribute node prior to child traversal.
        updated_node: Attribute node after child traversal.

    Returns:
        Transformed expression node.
    """
    return self.attribute_simplifier.rewrite_attribute(original_node, updated_node)

  def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.Import | cst.RemovalSentinel:
    """Delegate ``cst.Import`` rewriting and pruning to ``import_rewriter``.

    Args:
        original_node: Import node prior to child traversal.
        updated_node: Import node after child traversal.

    Returns:
        Rewritten ``cst.Import`` or ``cst.RemoveFromParent()``.
    """
    return self.import_rewriter.rewrite_import(original_node, updated_node)

  def leave_ImportFrom(
    self,
    original_node: cst.ImportFrom,
    updated_node: cst.ImportFrom,
  ) -> cst.ImportFrom | cst.Import | cst.RemovalSentinel:
    """Delegate ``cst.ImportFrom`` rewriting and pruning to ``import_rewriter``.

    Args:
        original_node: ImportFrom node prior to child traversal.
        updated_node: ImportFrom node after child traversal.

    Returns:
        Rewritten ``cst.ImportFrom`` / ``cst.Import`` or ``cst.RemoveFromParent()``.
    """
    return self.import_rewriter.rewrite_import_from(original_node, updated_node)

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """Delegate missing top-level import injection and deduplication to ``import_injector``.

    Args:
        original_node: Module node prior to child traversal.
        updated_node: Module node after child traversal.

    Returns:
        Modified ``cst.Module`` with injected and deduplicated imports.
    """
    return self.import_injector.inject_module_imports(original_node, updated_node)


__all__ = [
  "AttributeAliasSimplifier",
  "ImportFixer",
  "ImportFixerState",
  "ImportInjector",
  "ImportReq",
  "ImportResolver",
  "ImportStatementRewriter",
  "ResolutionPlan",
]
