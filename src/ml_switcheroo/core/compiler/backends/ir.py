"""Intermediate Representation (IR) Compiler Backend.

This module serializes LogicalGraph instances into canonical JSON format or
executable Python code conforming to the `ml_switcheroo_ir` specification.
"""

from typing import Any, Dict, List, Optional
import json

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import (
  LogicalGraph,
  topological_sort,
)


class IrBackend(CompilerBackend):
  """Compiler backend emitting Intermediate Representation (IR) artifacts.

  Serializes the input LogicalGraph into either a canonical JSON schema
  or programmatic Python CST statements using `ml_switcheroo_ir`.

  Attributes:
      semantics: Optional semantics manager for operator and attribute resolution.
      format: Target output format, either 'json' or 'python'.
  """

  def __init__(
    self,
    semantics: Optional[Any] = None,
    format: str = "json",
  ) -> None:
    """Initialize the IR backend with semantics and format settings.

    Args:
        semantics: Optional semantics manager or configuration.
        format: Serialization format ('json' or 'python'). Defaults to 'json'.
    """
    self.semantics = semantics
    self.format = format

  def compile(self, graph: LogicalGraph) -> str:
    """Compile a LogicalGraph into serialized IR text.

    Args:
        graph: The intermediate representation graph to serialize.

    Returns:
        str: Serialized JSON or Python code representing the computation graph.

    Raises:
        ValueError: If graph validation fails or invalid format is configured.
    """
    self._validate_graph(graph)

    if self.format == "python":
      return self.compile_to_python(graph)
    return self.compile_to_json(graph)

  def _validate_graph(self, graph: LogicalGraph) -> None:
    """Validate graph topology and structure before compilation.

    Args:
        graph: The LogicalGraph to validate.

    Raises:
        ValueError: If a dangling edge or node collision is detected.
    """
    if not isinstance(graph, LogicalGraph):
      raise ValueError("Input graph must be an instance of LogicalGraph.")

    node_ids = set(graph.nodes.keys())
    all_edges = list(graph.edges)
    pending = getattr(graph, "_pending_edges", [])
    if pending:
      all_edges.extend(pending)

    for edge in all_edges:
      if edge.source not in node_ids:
        raise ValueError(f"Dangling edge source '{edge.source}' not found in graph nodes.")
      if edge.target not in node_ids:
        raise ValueError(f"Dangling edge target '{edge.target}' not found in graph nodes.")

  def compile_to_json(self, graph: LogicalGraph) -> str:
    """Serialize a LogicalGraph into canonical, deterministic JSON.

    Args:
        graph: The LogicalGraph to serialize.

    Returns:
        str: Indented and sorted JSON string.
    """
    sorted_nodes = topological_sort(graph)

    # Build node-level inputs map from edges
    inputs_map: Dict[str, List[str]] = {n.id: [] for n in sorted_nodes}
    for edge in graph.edges:
      inputs_map[edge.target].append(edge.source)

    nodes_payload: List[Dict[str, Any]] = []
    for node in sorted_nodes:
      op_type = getattr(node, "op_type", None) or getattr(node, "kind", "")
      node_attrs = getattr(node, "attributes", {})
      node_dict: Dict[str, Any] = {
        "id": node.id,
        "kind": op_type,
        "op_type": op_type,
        "domain": getattr(node, "domain", "ai.onnx"),
        "version": getattr(node, "version", 1),
        "metadata": dict(sorted(node_attrs.items())),
        "attributes": dict(sorted(node_attrs.items())),
        "inputs": sorted(inputs_map.get(node.id, [])),
      }
      if node.sharding is not None:
        node_dict["sharding"] = {
          "axes": list(node.sharding.axes),
        }
      nodes_payload.append(node_dict)

    edges_payload: List[Dict[str, str]] = [
      {"source": edge.source, "target": edge.target} for edge in sorted(graph.edges, key=lambda e: (e.source, e.target))
    ]

    payload: Dict[str, Any] = {
      "name": graph.name,
      "nodes": nodes_payload,
      "edges": edges_payload,
    }

    if graph.mesh is not None:
      payload["mesh"] = {
        "shape": dict(sorted(graph.mesh.shape.items())),
      }

    return json.dumps(payload, indent=2, sort_keys=True)

  def compile_to_python(self, graph: LogicalGraph) -> str:
    """Emit Python code constructing the LogicalGraph via ml_switcheroo_ir.

    Args:
        graph: The LogicalGraph to compile to Python code.

    Returns:
        str: Valid Python source code building the graph.
    """
    lines: List[str] = [
      '"""Auto-generated LogicalGraph definition."""',
      "",
      "import ml_switcheroo_ir as sw_ir",
      "",
      "def build_graph() -> sw_ir.LogicalGraph:",
      '    """Construct and return the LogicalGraph."""',
    ]

    if graph.mesh is not None:
      mesh_repr = repr(dict(sorted(graph.mesh.shape.items())))
      lines.append(f"    mesh = sw_ir.LogicalMesh(shape={mesh_repr})")
    else:
      lines.append("    mesh = None")

    lines.append("    nodes = {")
    for node in topological_sort(graph):
      op_type = getattr(node, "op_type", None) or getattr(node, "kind", "")
      node_attrs = getattr(node, "attributes", {})
      meta_str = repr(dict(sorted(node_attrs.items())))
      sharding_str = "None"
      if node.sharding is not None:
        sharding_str = f"sw_ir.PartitionSpec(axes={repr(node.sharding.axes)})"

      lines.append(f"        {repr(node.id)}: sw_ir.LogicalNode(")
      lines.append(f"            id={repr(node.id)},")
      lines.append(f"            op_type={repr(op_type)},")
      lines.append(f"            domain={repr(getattr(node, 'domain', 'ai.onnx'))},")
      lines.append(f"            version={getattr(node, 'version', 1)},")
      lines.append(f"            attributes={meta_str},")
      lines.append(f"            sharding={sharding_str},")
      lines.append("        ),")
    lines.append("    }")

    lines.append("    edges = [")
    for edge in sorted(graph.edges, key=lambda e: (e.source, e.target)):
      lines.append(f"        sw_ir.LogicalEdge(source={repr(edge.source)}, target={repr(edge.target)}),")
    lines.append("    ]")

    lines.append("    return sw_ir.LogicalGraph(")
    lines.append(f"        name={repr(graph.name)},")
    lines.append("        nodes=nodes,")
    lines.append("        edges=edges,")
    lines.append("        mesh=mesh,")
    lines.append("    )")
    lines.append("")

    return "\n".join(lines)
