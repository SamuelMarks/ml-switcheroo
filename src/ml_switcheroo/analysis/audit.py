"""
Audit functionality to determine coverage gaps for a source codebase.

This module provides the :class:`CoverageScanner`, a static analysis visitor
that inspects source code to identify API usage (calls and attributes) relative
to specific frameworks (e.g., PyTorch, JAX). It resolves import aliases to
fully qualified names (FQNs) and checks them against the :class:`SemanticsManager`
to determine if they are currently supported by the transpiler.
"""

import libcst as cst
from typing import Dict, Set, Tuple

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.scanners import get_full_name


class CoverageScanner(cst.CSTVisitor):
  """
  Scans a file to identify API calls and checks if they exist in the Semantics Manager.

  It tracks import aliases to correctly resolve calls like `jnp.sum` back to
  `jax.numpy.sum` before querying the knowledge base.
  """

  results: Dict[str, Tuple[bool, str]]
  """ 
    A dictionary mapping Fully Qualified Names (FQN) 
    to a tuple of `(is_supported, framework_key)`. 
    """

  def __init__(self, semantics: SemanticsManager, allowed_roots: Set[str]):
    """
    Initializes the scanner.

    Args:
        semantics: The knowledge base manager used to verify support.
        allowed_roots: A set of framework root strings to consider (e.g. `{'torch', 'jax'}`).
                       APIs not starting with these roots are ignored.
    """
    self.semantics = semantics
    self.allowed_roots = allowed_roots

    # Maps local alias -> Full Path (e.g., 'jnp' -> 'jax.numpy')
    self._alias_map: Dict[str, str] = {}

    # Results: Map FQN -> (IsSupported, FrameworkKey)
    self.results = {}

  def visit_Import(self, node: cst.Import) -> None:
    """
    Visits `import ...` statements to populate the alias map.

    Args:
        node: The CST import node.
    """
    for alias in node.names:
      full_name = get_full_name(alias.name)
      local_name = alias.asname.name.value if alias.asname else full_name.split(".")[0]
      self._alias_map[local_name] = full_name

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """
    Visits `from ... import ...` statements to populate the alias map.

    Args:
        node: The CST import-from node.
    """
    if not node.module:
      return
    module_name = get_full_name(node.module)

    for alias in node.names:
      if isinstance(alias, cst.ImportAlias):
        import_name = alias.name.value
        local_name = alias.asname.name.value if alias.asname else import_name
        full_path = f"{module_name}.{import_name}"
        self._alias_map[local_name] = full_path

  def visit_Call(self, node: cst.Call) -> None:
    """
    Visits function call nodes to check the invoked function.

    Args:
        node: The CST call node.
    """
    self._check_node(node.func)

  def visit_Attribute(self, node: cst.Attribute) -> None:
    """
    Visits attributes to check for framework constants.

    This allows detecting usage like `torch.float32` which are not calls
    but still require support. Checks are idempotent via the results dictionary.

    Args:
        node: The CST attribute node.
    """
    self._check_node(node)

  def _check_node(self, node: cst.CSTNode) -> None:
    """
    Resolves the FQN of a node and records its support status.

    Args:
        node: The AST node to inspect (Name or Attribute).
    """
    fqn = self._resolve_fqn(node)
    if not fqn:
      return

    # Check Root filtering
    root = fqn.split(".")[0]
    if root not in self.allowed_roots:
      return

    # Identify Status
    definition = self.semantics.get_definition(fqn)
    is_supported = definition is not None

    # Identify Framework
    # If supported, use the variant info from Semantics to get exact framework Key.
    # Else guess from root.
    framework = root

    if definition:
      # definition is (abstract_id, details_dict)
      _, details = definition
      variants = details.get("variants", {})
      for fw_key, vari in variants.items():
        if vari and vari.get("api") == fqn:
          framework = fw_key
          break

    self.results[fqn] = (is_supported, framework)

  def _resolve_fqn(self, node: cst.CSTNode) -> str:
    """
    Resolves a CST node to its Fully Qualified Name string.

    Applies alias rewriting (e.g. `jnp.abs` -> `jax.numpy.abs`) based on
    imports found earlier in the file.

    Args:
        node: The node to resolve.

    Returns:
        str: The resolved fully qualified name, or empty string if unresolvable.
    """
    # 1. Flatten CST to string
    raw_name = get_full_name(node)
    if not raw_name:
      return ""

    # 2. Resolve Alias
    parts = raw_name.split(".")
    root = parts[0]

    if root in self._alias_map:
      resolved_root = self._alias_map[root]
      if len(parts) > 1:
        return f"{resolved_root}.{'.'.join(parts[1:])}"
      return resolved_root

    return raw_name
