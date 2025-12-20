"""
Audit functionality to determine coverage gaps for a source codebase.
"""

import libcst as cst
from typing import Dict, Set, List, Tuple, Optional

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.scanners import get_full_name


class CoverageScanner(cst.CSTVisitor):
  """
  Scans a file to identify API calls and checks if they exist in the Semantics Manager.

  Attributes:
      results (Dict): FQN -> (IsSupported: bool, FrameworkKey: str)
  """

  def __init__(self, semantics: SemanticsManager, allowed_roots: Set[str]):
    self.semantics = semantics
    self.allowed_roots = allowed_roots

    # Maps local alias -> Full Path (e.g., 'jnp' -> 'jax.numpy')
    self._alias_map: Dict[str, str] = {}

    # Results: Map FQN -> (IsSupported, FrameworkKey)
    self.results: Dict[str, Tuple[bool, str]] = {}

  def visit_Import(self, node: cst.Import) -> None:
    for alias in node.names:
      full_name = get_full_name(alias.name)
      local_name = alias.asname.name.value if alias.asname else full_name.split(".")[0]
      self._alias_map[local_name] = full_name

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
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
    self._check_node(node.func)

  def visit_Attribute(self, node: cst.Attribute) -> None:
    # Check attributes acting as constants (e.g., torch.float32)
    # We assume _check_node is idempotent via dict storage.
    self._check_node(node)

  def _check_node(self, node: cst.CSTNode):
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
