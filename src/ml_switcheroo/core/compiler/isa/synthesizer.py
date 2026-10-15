"""Unified ISA Synthesizer."""

import json
import os
from typing import Any, Dict, List

import yaml

from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator


class IsaSynthesizer:
  """Unified Base Class for Hardware ISA Synthesizers."""

  def __init__(
    self,
    semantics: Any,
    allocator: GenericRegisterAllocator[Any],
    macros_yaml_path: str,
    macros_json_path: str,
    macro_module: Any,
  ) -> None:
    """Initialize the ISA synthesizer with semantics, register allocator, and macro mappings.

    Args:
        semantics: SemanticsManager instance for resolving abstract operations.
        allocator: Register allocator managing physical register assignments.
        macros_yaml_path: Filesystem path to the YAML macro registry file.
        macros_json_path: Fallback filesystem path to the JSON macro registry file.
        macro_module: Python module containing macro expansion functions.
    """
    self.semantics = semantics
    self.allocator = allocator

    self.macro_registry = {}
    mapping: Dict[str, Any] = {}
    if os.path.exists(macros_yaml_path):
      try:
        with open(macros_yaml_path, "r", encoding="utf-8") as f:
          mapping = yaml.safe_load(f) or {}
      except FileNotFoundError:
        pass

    if not mapping and os.path.exists(macros_json_path):
      try:
        with open(macros_json_path, "r", encoding="utf-8") as f:
          mapping = json.load(f)
      except FileNotFoundError:
        pass

    for key, func_name in mapping.items():
      if hasattr(macro_module, func_name):
        self.macro_registry[key] = getattr(macro_module, func_name)

  def _resolve_abstract_id(self, op_type: str) -> str:
    """Resolve an operation type string to its canonical abstract operation ID.

    Args:
        op_type: Operation type or qualified API name.

    Returns:
        Resolved abstract operation ID, or ``op_type`` if no definition is found.
    """
    defn = self.semantics.get_definition(op_type)
    if not defn and ("." in op_type):
      suffix = op_type.split(".", 1)[-1]
      defn = self.semantics.get_definition(suffix)
    return defn[0] if defn else op_type

  def _get_input_map(self, graph: LogicalGraph) -> Dict[str, List[str]]:
    """Build an input adjacency map from target node IDs to ordered source node IDs.

    Args:
        graph: The LogicalGraph to inspect.

    Returns:
        Dictionary mapping each target node ID to a list of incoming source node IDs.
    """
    input_map: Dict[str, List[str]] = {}
    for edge in graph.edges:
      if edge.target not in input_map:
        input_map[edge.target] = []
      input_map[edge.target].append(edge.source)
    return input_map
