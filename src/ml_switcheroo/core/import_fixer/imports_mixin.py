"""Import Statement Rewriter Collaborator.

Handles visiting, cleaning, and rewriting ``cst.Import`` and ``cst.ImportFrom``
nodes based on the centralized ``ResolutionPlan`` stored in ``ImportFixerState``.
"""

import libcst as cst

from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.resolution import ImportReq
from ml_switcheroo.core.import_fixer.utils import create_dotted_name
from ml_switcheroo.core.scanners import get_full_name


class ImportStatementRewriter:
  """Collaborator rewriting and pruning ``Import`` and ``ImportFrom`` CST statements.

  Attributes:
      state: Shared ``ImportFixerState`` instance.
  """

  def __init__(self, state: ImportFixerState) -> None:
    """Initialize the import statement rewriter with shared ``ImportFixerState``.

    Args:
        state: Shared ``ImportFixerState`` instance.
    """
    self.state: ImportFixerState = state

  def make_alias_node(self, req: ImportReq) -> cst.ImportAlias:
    """Construct a CST ``ImportAlias`` from an ``ImportReq``.

    Args:
        req: Target import requirement details.

    Returns:
        Constructed ``cst.ImportAlias`` node.
    """
    name_str = f"{req.module}.{req.subcomponent}" if req.subcomponent else req.module

    asname_node = None
    should_alias = False

    if req.alias:
      leaf = req.subcomponent if req.subcomponent else req.module.split(".")[-1]
      if req.alias != leaf or "." in name_str:
        should_alias = True

    if should_alias and req.alias:
      asname_node = cst.AsName(name=cst.Name(req.alias))

    return cst.ImportAlias(name=create_dotted_name(name_str), asname=asname_node)

  def rewrite_import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.Import | cst.RemovalSentinel:
    """Rewrite or prune an ``Import`` node using ``self.state``.

    Args:
        original_node: Original ``cst.Import`` node prior to traversal.
        updated_node: Updated ``cst.Import`` node.

    Returns:
        Rewritten ``cst.Import`` node or ``cst.RemoveFromParent()`` if all aliases are pruned.
    """
    new_aliases: list[cst.ImportAlias] = []
    plan = self.state.plan

    for alias in updated_node.names:
      full_name = get_full_name(alias.name)
      root_pkg = full_name.split(".")[0]

      if full_name in plan.mappings and root_pkg in self.state.source_frameworks:
        req = plan.mappings[full_name]
        new_alias = self.make_alias_node(req)

        if not req.alias and alias.asname and not new_alias.asname:
          new_alias = new_alias.with_changes(asname=alias.asname)

        new_aliases.append(new_alias)
        self.state.track_definition(new_alias)
        self.state.satisfied_injections.add(req.signature)
        continue

      self.state.track_definition(alias)

      for req in plan.required_imports:
        if req.module == full_name and not req.subcomponent:
          self.state.satisfied_injections.add(req.signature)

      bound_name = alias.asname.name.value if (alias.asname and isinstance(alias.asname.name, cst.Name)) else root_pkg
      if bound_name in self.state.used_names:
        new_aliases.append(alias)

    if not new_aliases:
      return cst.RemoveFromParent()

    last_alias = new_aliases[-1]
    if last_alias.comma != cst.MaybeSentinel.DEFAULT:
      new_aliases[-1] = last_alias.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return updated_node.with_changes(names=new_aliases)

  def rewrite_import_from(
    self,
    original_node: cst.ImportFrom,
    updated_node: cst.ImportFrom,
  ) -> cst.ImportFrom | cst.Import | cst.RemovalSentinel:
    """Rewrite or prune an ``ImportFrom`` node using ``self.state``.

    Args:
        original_node: Original ``cst.ImportFrom`` node prior to traversal.
        updated_node: Updated ``cst.ImportFrom`` node.

    Returns:
        Modified ``cst.ImportFrom`` or ``cst.Import`` node, or ``cst.RemoveFromParent()`` if pruned.
    """
    if not updated_node.module:
      return updated_node

    module_name = get_full_name(updated_node.module)
    root_pkg = module_name.split(".")[0]
    plan = self.state.plan

    if isinstance(updated_node.names, cst.ImportStar):
      if root_pkg in self.state.source_frameworks:
        return cst.RemoveFromParent()
      return updated_node

    if len(updated_node.names) == 1 and isinstance(updated_node.names[0], cst.ImportAlias):
      import_name = updated_node.names[0].name.value
      lookup_key = f"{module_name}.{import_name}"

      if lookup_key in plan.mappings and root_pkg in self.state.source_frameworks:
        req = plan.mappings[lookup_key]
        new_node = cst.Import(names=[self.make_alias_node(req)])
        self.state.satisfied_injections.add(req.signature)
        self.state.track_definition(new_node.names[0])
        return new_node

    new_aliases: list[cst.ImportAlias] = []
    for alias in updated_node.names:
      self.state.track_definition(alias)

      if alias.asname and isinstance(alias.asname.name, cst.Name):
        bound_name = alias.asname.name.value
      elif isinstance(alias.name, cst.Name):
        bound_name = alias.name.value
      else:
        bound_name = ""
      if bound_name in self.state.used_names:
        new_aliases.append(alias)

    if not new_aliases:
      return cst.RemoveFromParent()

    if not updated_node.rpar:
      last_alias = new_aliases[-1]
      if last_alias.comma != cst.MaybeSentinel.DEFAULT:
        new_aliases[-1] = last_alias.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return updated_node.with_changes(names=new_aliases)


__all__ = ["ImportStatementRewriter"]
