"""Intermediate Representation (IR) Frontend.

This module provides parsers and lifters to ingest Intermediate Representation
payloads (JSON or Python CST using `ml_switcheroo_ir`) into a `LogicalGraph`.
"""

from typing import Any, Dict, List, Optional, Set
import json
import libcst as cst

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.ir import (
  NodeDict,
  LogicalEdge,
  LogicalGraph,
  LogicalMesh,
  LogicalNode,
  PartitionSpec,
  topological_sort,
)


class IrParseError(ValueError):
  """Exception raised when an IR string or file fails syntactic or schema validation."""

  def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None) -> None:
    """Initialize the IrParseError with contextual line and column indicators.

    Args:
        message: Description of the syntactic or validation failure.
        line: 1-based line number where the error was detected.
        column: 1-based column offset where the error occurred.
    """
    loc = f" (line {line}, col {column})" if line is not None else ""
    super().__init__(f"{message}{loc}")
    self.line = line
    self.column = column


class IrJsonParser:
  """Parser for deserializing JSON-formatted Intermediate Representation schemas."""

  def parse(self, json_str: str) -> LogicalGraph:
    """Parse a JSON string into a LogicalGraph instance.

    Args:
        json_str: The raw JSON string representing the graph.

    Returns:
        LogicalGraph: The deserialized computation graph.

    Raises:
        IrParseError: If the JSON is syntactically invalid or misses required fields.
    """
    try:
      data: Dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
      raise IrParseError(f"Malformed JSON syntax: {e.msg}", line=e.lineno, column=e.colno) from e

    if not isinstance(data, dict):
      raise IrParseError("Root of IR JSON must be a dictionary object.")

    name = data.get("name", "Model")
    raw_nodes = data.get("nodes", [])

    nodes_dict: Dict[str, LogicalNode] = {}
    edges: List[LogicalEdge] = []

    # Handle nodes either as list or dict
    if isinstance(raw_nodes, dict):
      node_iterable = list(raw_nodes.values())
    elif isinstance(raw_nodes, list):
      node_iterable = raw_nodes
    else:
      raise IrParseError("'nodes' field must be a list or dictionary.")

    for node_data in node_iterable:
      if not isinstance(node_data, dict):
        raise IrParseError("Every node entry must be a dictionary.")

      node_id = node_data.get("id")
      if not node_id or not isinstance(node_id, str):
        raise IrParseError("Node is missing mandatory 'id' string field.")

      kind = node_data.get("kind") or node_data.get("op_type")
      if not kind or not isinstance(kind, str):
        raise IrParseError(f"Node '{node_id}' is missing mandatory 'kind' string field.")

      raw_meta = node_data.get("metadata") or node_data.get("attributes", {})
      meta_dict = {str(k): str(v) for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}

      sharding: Optional[PartitionSpec] = None
      if "sharding" in node_data and isinstance(node_data["sharding"], dict):
        axes = tuple(node_data["sharding"].get("axes", []))
        sharding = PartitionSpec(axes=axes)

      node = LogicalNode(
        id=node_id,
        op_type=kind,
        attributes=meta_dict,
        sharding=sharding,
      )
      # Store domain and version if present
      if "domain" in node_data:
        setattr(node, "domain", str(node_data["domain"]))
      if "version" in node_data:
        setattr(node, "version", int(node_data["version"]))

      nodes_dict[node_id] = node

      # Derive implicit edges from node inputs if specified
      if "inputs" in node_data and isinstance(node_data["inputs"], list):
        for inp_id in node_data["inputs"]:
          if isinstance(inp_id, str):
            edges.append(LogicalEdge(source=inp_id, target=node_id))

    # Read explicit edges if present, avoiding duplicates
    existing_edges: Set[tuple[str, str]] = {(e.source, e.target) for e in edges}
    for edge_data in data.get("edges", []):
      if isinstance(edge_data, dict) and "source" in edge_data and "target" in edge_data:
        pair = (str(edge_data["source"]), str(edge_data["target"]))
        if pair not in existing_edges:
          edges.append(LogicalEdge(source=pair[0], target=pair[1]))
          existing_edges.add(pair)

    mesh: Optional[LogicalMesh] = None
    if "mesh" in data and isinstance(data["mesh"], dict):
      shape_dict = {str(k): int(v) for k, v in data["mesh"].get("shape", {}).items()}
      mesh = LogicalMesh(shape=shape_dict)

    return LogicalGraph(
      name=name,
      nodes=nodes_dict,
      edges=edges,
      mesh=mesh,
    )


class IrPythonParser:
  """Parser for extracting LogicalGraph instances from Python CST or code."""

  def parse(self, code: str) -> LogicalGraph:
    """Parse Python source code into a LogicalGraph.

    Args:
        code: Python source code string.

    Returns:
        LogicalGraph: The extracted computational graph.
    """
    # First try PythonFrontend to extract classes / layers
    frontend = PythonFrontend(code)
    graph = frontend.parse_to_graph()
    if graph.nodes:
      return graph

    # Fallback to empty graph
    return LogicalGraph()


class IrLifter:
  """Lifter and normalizer for validating topology and ordering nodes."""

  def lift(self, graph: LogicalGraph) -> LogicalGraph:
    """Normalize nodes, eliminate dangling edges, and order topologically.

    Args:
        graph: Raw input graph.

    Returns:
        LogicalGraph: Topologically sorted and normalized graph.

    Raises:
        IrParseError: If cycle or dangling reference is found.
    """
    node_ids = set(graph.nodes.keys())
    valid_edges: List[LogicalEdge] = []

    all_edges = list(graph.edges)
    pending = getattr(graph, "_pending_edges", [])
    if pending:
      all_edges.extend(pending)

    for edge in all_edges:
      if edge.source not in node_ids:
        raise IrParseError(f"Dangling edge source '{edge.source}' not found in graph.")
      if edge.target not in node_ids:
        raise IrParseError(f"Dangling edge target '{edge.target}' not found in graph.")
      valid_edges.append(edge)

    for node in graph.nodes.values():
      node.inputs = []
    for edge in valid_edges:
      if edge.source not in graph.nodes[edge.target].inputs:
        graph.nodes[edge.target].inputs.append(edge.source)

    sorted_nodes = topological_sort(graph)
    graph.nodes = NodeDict(graph, {n.id: n for n in sorted_nodes})
    return graph


class IrToCstGenerator:
  """Bridge synthesizing a LibCST Module from a LogicalGraph."""

  def generate(self, graph: LogicalGraph) -> cst.Module:
    """Generate a LibCST Module representing the computation in the LogicalGraph.

    Args:
        graph: The source LogicalGraph.

    Returns:
        cst.Module: Synthesized LibCST AST representing a neural module class.
    """
    sorted_nodes = topological_sort(graph)

    # Collect inputs map
    inputs_map: Dict[str, List[str]] = {n.id: [] for n in sorted_nodes}
    for edge in graph.edges:
      inputs_map[edge.target].append(edge.source)

    init_stmts: List[cst.BaseStatement] = [
      cst.SimpleStatementLine(
        [
          cst.Expr(
            cst.Call(
              func=cst.Attribute(
                value=cst.Call(func=cst.Name("super"), args=[]),
                attr=cst.Name("__init__"),
              ),
              args=[],
            )
          )
        ]
      )
    ]

    forward_stmts: List[cst.BaseStatement] = []
    for node in sorted_nodes:
      op_type: str = str(getattr(node, "op_type", None) or getattr(node, "kind", ""))
      if op_type.lower() == "input":
        continue

      # Create layer attribute in __init__
      init_stmts.append(
        cst.SimpleStatementLine(
          [
            cst.Assign(
              targets=[cst.AssignTarget(cst.Attribute(value=cst.Name("self"), attr=cst.Name(node.id)))],
              value=cst.Call(
                func=cst.Attribute(value=cst.Name("nn"), attr=cst.Name(op_type)),
                args=[],
              ),
            )
          ]
        )
      )

      # Create forward call
      inp_vars = inputs_map.get(node.id, [])
      call_arg = cst.Name(inp_vars[0]) if inp_vars else cst.Name("x")
      forward_stmts.append(
        cst.SimpleStatementLine(
          [
            cst.Assign(
              targets=[cst.AssignTarget(cst.Name(node.id))],
              value=cst.Call(
                func=cst.Attribute(value=cst.Name("self"), attr=cst.Name(node.id)),
                args=[cst.Arg(value=call_arg)],
              ),
            )
          ]
        )
      )

    last_id = sorted_nodes[-1].id if sorted_nodes else "x"
    forward_stmts.append(cst.SimpleStatementLine([cst.Return(value=cst.Name(last_id))]))

    class_def = cst.ClassDef(
      name=cst.Name(graph.name or "GeneratedNet"),
      bases=[cst.Arg(value=cst.Attribute(value=cst.Name("nn"), attr=cst.Name("Module")))],
      body=cst.IndentedBlock(
        body=[
          cst.FunctionDef(
            name=cst.Name("__init__"),
            params=cst.Parameters(params=[cst.Param(name=cst.Name("self"))]),
            body=cst.IndentedBlock(body=init_stmts),
          ),
          cst.FunctionDef(
            name=cst.Name("forward"),
            params=cst.Parameters(params=[cst.Param(name=cst.Name("self")), cst.Param(name=cst.Name("x"))]),
            body=cst.IndentedBlock(body=forward_stmts),
          ),
        ]
      ),
    )

    return cst.Module(
      body=[
        cst.SimpleStatementLine(
          [
            cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))]),
            cst.ImportFrom(
              module=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")),
              names=[cst.ImportAlias(name=cst.Name("nn"))],
            ),
          ]
        ),
        class_def,
      ]
    )


class IrFrontend:
  """Compiler frontend that ingests IR JSON or Python code into a LogicalGraph."""

  def __init__(self, code: str) -> None:
    """Initialize the IR frontend with input source code or JSON.

    Args:
        code: Input JSON string or Python code representation.
    """
    self.code = code

  def parse_to_graph(self, code: Optional[str] = None) -> LogicalGraph:
    """Parse the source payload into a normalized LogicalGraph.

    Args:
        code: Optional override for the source code payload.

    Returns:
        LogicalGraph: The parsed and normalized computation graph.
    """
    src = (code or self.code).strip()
    if not src:
      return LogicalGraph()

    if src.startswith("{") or src.startswith("["):
      parser = IrJsonParser()
      graph = parser.parse(src)
    else:
      parser_py = IrPythonParser()
      graph = parser_py.parse(src)

    lifter = IrLifter()
    return lifter.lift(graph)
