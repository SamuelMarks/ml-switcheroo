"""Intermediate Representation (IR) Frontend.

This module provides parsers and lifters to ingest Intermediate Representation
payloads (JSON or Python CST using `ml_switcheroo_ir`) into a `LogicalGraph`.
"""

import ast
import json
import types
import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.ir import (
  LogicalEdge,
  LogicalGraph,
  LogicalMesh,
  LogicalNode,
  PartitionSpec,
  topological_sort,
)
from ml_switcheroo.exceptions import SyntaxParseError


class IrParseError(SyntaxParseError):
  """Exception raised when an IR string or file fails syntactic or schema validation."""

  def __init__(self, message: str, line: int | None = None, column: int | None = None) -> None:
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

  @staticmethod
  def _parse_sharding(raw_sharding: JsonValue) -> PartitionSpec | None:
    """Deserialize an optional ``sharding`` JSON object into a ``PartitionSpec``.

    Args:
        raw_sharding: Raw JSON ``sharding`` field value.

    Returns:
        Constructed ``PartitionSpec`` if ``raw_sharding`` is a dictionary, otherwise ``None``.
    """
    if not isinstance(raw_sharding, dict):
      return None
    raw_axes = raw_sharding.get("axes", [])
    axes_seq = raw_axes if isinstance(raw_axes, list) else []
    axes: list[str | tuple[str, ...] | None] = [
      tuple(str(x) for x in ax) if isinstance(ax, (list, tuple)) else (str(ax) if isinstance(ax, str) else None)
      for ax in axes_seq
    ]
    return PartitionSpec(axes=tuple(axes))

  @classmethod
  def _parse_node_entry(cls, node_data: JsonValue) -> tuple[LogicalNode, list[LogicalEdge]]:
    """Deserialize a single JSON node entry and its implicit ``inputs`` edges.

    Args:
        node_data: Raw JSON node dictionary.

    Returns:
        Tuple of ``(LogicalNode, implicit_edges)``.

    Raises:
        IrParseError: If ``node_data`` is not a dictionary or lacks mandatory ``id``/``kind`` fields.
    """
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
    domain = str(node_data["domain"]) if "domain" in node_data else "ai.onnx"
    raw_version = node_data.get("version")
    version = int(raw_version) if isinstance(raw_version, (int, float, str)) else 1

    node = LogicalNode(
      id=node_id,
      op_type=kind,
      attributes=meta_dict,
      sharding=cls._parse_sharding(node_data.get("sharding")),
      domain=domain,
      version=version,
    )

    raw_inputs = node_data.get("inputs")
    implicit_edges = (
      [LogicalEdge(source=inp_id, target=node_id) for inp_id in raw_inputs if isinstance(inp_id, str)]
      if isinstance(raw_inputs, list)
      else []
    )
    return node, implicit_edges

  @staticmethod
  def _merge_explicit_edges(edges: list[LogicalEdge], raw_edges: JsonValue) -> None:
    """Append deduplicated explicit ``edges`` JSON entries into ``edges``.

    Args:
        edges: Mutable list of ``LogicalEdge`` instances seeded from implicit ``inputs``.
        raw_edges: Raw ``edges`` JSON value from the root payload.
    """
    existing_edges: set[tuple[str, str]] = {(e.source, e.target) for e in edges}
    edges_seq = raw_edges if isinstance(raw_edges, list) else []
    for edge_data in edges_seq:
      if isinstance(edge_data, dict) and "source" in edge_data and "target" in edge_data:
        pair = (str(edge_data["source"]), str(edge_data["target"]))
        if pair not in existing_edges:
          edges.append(LogicalEdge(source=pair[0], target=pair[1]))
          existing_edges.add(pair)

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
      data: JsonValue = json.loads(json_str)
    except json.JSONDecodeError as e:
      raise IrParseError(f"Malformed JSON syntax: {e.msg}", line=e.lineno, column=e.colno) from e

    if not isinstance(data, dict):
      raise IrParseError("Root of IR JSON must be a dictionary object.")

    raw_name = data.get("name", "Model")
    name = str(raw_name) if isinstance(raw_name, str) else "Model"
    raw_nodes = data.get("nodes", [])

    if isinstance(raw_nodes, dict):
      node_iterable = list(raw_nodes.values())
    elif isinstance(raw_nodes, list):
      node_iterable = raw_nodes
    else:
      raise IrParseError("'nodes' field must be a list or dictionary.")

    nodes_dict: dict[str, LogicalNode] = {}
    edges: list[LogicalEdge] = []
    for node_data in node_iterable:
      node, implicit_edges = self._parse_node_entry(node_data)
      nodes_dict[node.id] = node
      edges.extend(implicit_edges)

    self._merge_explicit_edges(edges, data.get("edges", []))

    mesh: LogicalMesh | None = None
    raw_mesh = data.get("mesh")
    if isinstance(raw_mesh, dict):
      raw_shape = raw_mesh.get("shape", {})
      shape_map = raw_shape if isinstance(raw_shape, dict) else {}
      mesh = LogicalMesh(shape={str(k): int(v) for k, v in shape_map.items() if isinstance(v, (int, float, str))})

    return LogicalGraph(name=name, nodes=nodes_dict, edges=edges, mesh=mesh)


class _CapturedGraphFn:
  """Zero-argument captured AST function wrapper for `IrPythonParser`."""

  def __init__(self, fn_ip: typing.Callable[[], "IrEvalValue"]) -> None:
    """Initialize wrapper around a zero-argument evaluation thunk.

    Args:
        fn_ip: Zero-argument evaluation callable.
    """
    self._fn_ip = fn_ip

  def invoke(self) -> "IrEvalValue":
    """Execute the captured function body.

    Returns:
        Evaluated IR value.
    """
    return self._fn_ip()


IrEvalValue: typing.TypeAlias = (
  LogicalGraph
  | LogicalNode
  | LogicalEdge
  | LogicalMesh
  | PartitionSpec
  | types.ModuleType
  | _CapturedGraphFn
  | type
  | types.FunctionType
  | types.BuiltinFunctionType
  | bytes
  | complex
  | types.EllipsisType
  | JsonValue
  | tuple["IrEvalValue", ...]
  | list["IrEvalValue"]
  | dict[str, "IrEvalValue"]
)


@typing.runtime_checkable
class _IrCallableProtocol(typing.Protocol):
  """Protocol for IR constructors and callables evaluated in `IrPythonParser`."""

  def __call__(self, *args: "IrEvalValue", **kwargs: "IrEvalValue") -> "IrEvalValue":
    """Invoke IR constructor or helper function.

    Args:
        *args: Positional evaluated AST arguments.
        **kwargs: Keyword evaluated AST arguments.

    Returns:
        Constructed IR object or value.
    """


@typing.runtime_checkable
class _PendingEdgesProtocol(typing.Protocol):
  """Protocol for `LogicalGraph` instances carrying deferred `_pending_edges`.

  Attributes:
      _pending_edges: Deferred edges accumulated prior to topological lifting.
  """

  _pending_edges: list[LogicalEdge]


_IR_MODULE_EXPORTS: dict[str, IrEvalValue] = {
  "LogicalGraph": LogicalGraph,
  "LogicalNode": LogicalNode,
  "LogicalEdge": LogicalEdge,
  "LogicalMesh": LogicalMesh,
  "PartitionSpec": PartitionSpec,
}


def _is_ir_callable(val: IrEvalValue) -> typing.TypeGuard[_IrCallableProtocol]:
  """Check whether ``val`` is an evaluated IR constructor or helper callable.

  Args:
      val: Evaluated AST expression value.

  Returns:
      True if ``val`` is callable.
  """
  return callable(val)


class IrPythonParser:
  """Parser for extracting LogicalGraph instances from Python CST or code."""

  @staticmethod
  def _eval_ast_expr(node: ast.AST, env: dict[str, IrEvalValue]) -> IrEvalValue:
    """Evaluate a simple AST expression generated by ``IrBackend.compile_to_python``.

    Args:
        node: Python ``ast.AST`` expression node.
        env: Symbol dictionary mapping names (`sw_ir`, `nodes`, `edges`, `mesh`) to values.

    Returns:
        Evaluated Python object (e.g. ``LogicalGraph``, ``LogicalNode``, ``LogicalEdge``, ``LogicalMesh``).

    Raises:
        IrParseError: If ``node`` is an unsupported AST construct or invokes a non-callable value.
    """
    if isinstance(node, ast.Constant):
      val: IrEvalValue = node.value
      return val
    if isinstance(node, ast.Name):
      return env[node.id]
    if isinstance(node, ast.Dict):
      return {
        str(IrPythonParser._eval_ast_expr(k, env)): IrPythonParser._eval_ast_expr(v, env)
        for k, v in zip(node.keys, node.values, strict=False)
        if k is not None
      }
    if isinstance(node, ast.List):
      return [IrPythonParser._eval_ast_expr(elt, env) for elt in node.elts]
    if isinstance(node, ast.Attribute):
      IrPythonParser._eval_ast_expr(node.value, env)
      return _IR_MODULE_EXPORTS[node.attr]
    if not isinstance(node, ast.Call):
      raise IrParseError(f"Unsupported IR AST expression: {type(node).__name__}")
    func_val = IrPythonParser._eval_ast_expr(node.func, env)
    if not _is_ir_callable(func_val):
      raise IrParseError(f"Non-callable target in Python IR expression: {func_val!r}")
    pos_args = [IrPythonParser._eval_ast_expr(a, env) for a in node.args]
    kw_args = {kw.arg: IrPythonParser._eval_ast_expr(kw.value, env) for kw in node.keywords if kw.arg is not None}
    return func_val(*pos_args, **kw_args)

  @classmethod
  def _exec_ast_body(cls, stmts: list[ast.stmt], env: dict[str, IrEvalValue]) -> IrEvalValue:
    """Execute a sequence of simple AST statements (`Assign`, `Return`, `FunctionDef`, `Expr`).

    Args:
        stmts: List of Python ``ast.stmt`` nodes.
        env: Mutable environment dictionary.

    Returns:
        Return value if a ``Return`` statement is encountered, otherwise ``None``.
    """
    for stmt in stmts:
      if isinstance(stmt, ast.Assign):
        val = cls._eval_ast_expr(stmt.value, env)
        for tgt in stmt.targets:
          if isinstance(tgt, ast.Name):
            env[tgt.id] = val
      elif isinstance(stmt, ast.Return) and stmt.value is not None:
        return cls._eval_ast_expr(stmt.value, env)
      elif isinstance(stmt, ast.FunctionDef):
        fn_body = list(stmt.body)

        def _fn(_body: list[ast.stmt] = fn_body) -> IrEvalValue:
          """Execute captured function body.

          Args:
              _body: Bound AST statement list for the function body.

          Returns:
              Evaluated function return value.
          """
          return cls._exec_ast_body(_body, dict(env))

        env[stmt.name] = _CapturedGraphFn(_fn)
    return None

  def parse(self, code: str) -> LogicalGraph:
    """Parse Python source code into a LogicalGraph.

    Args:
        code: Python source code string.

    Returns:
        The extracted LogicalGraph.
    """
    if "ml_switcheroo_ir" in code or "LogicalGraph" in code:
      import ml_switcheroo_ir as sw_ir

      tree = ast.parse(code)
      env: dict[str, IrEvalValue] = {
        "sw_ir": sw_ir,
        "ml_switcheroo_ir": sw_ir,
      }
      self._exec_ast_body(list(tree.body), env)
      bg_fn = env.get("build_graph")
      if isinstance(bg_fn, _CapturedGraphFn):
        res = bg_fn.invoke()
        if isinstance(res, LogicalGraph):
          return res
      for val in env.values():
        if isinstance(val, LogicalGraph):
          return val

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
    valid_edges: list[LogicalEdge] = []

    all_edges = list(graph.edges)
    if isinstance(graph, _PendingEdgesProtocol) and isinstance(graph._pending_edges, list):
      all_edges.extend(graph._pending_edges)

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
    graph.nodes.clear()
    graph.nodes.update({n.id: n for n in sorted_nodes})
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
    inputs_map: dict[str, list[str]] = {n.id: [] for n in sorted_nodes}
    for edge in graph.edges:
      inputs_map[edge.target].append(edge.source)

    init_stmts: list[cst.BaseStatement] = [
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

    forward_stmts: list[cst.BaseStatement] = []
    for node in sorted_nodes:
      op_type: str = node.op_type
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

    from ml_switcheroo.config import default_source_framework

    root_pkg = default_source_framework()
    return cst.Module(
      body=[
        cst.SimpleStatementLine(
          [
            cst.Import(names=[cst.ImportAlias(name=cst.Name(root_pkg))]),
            cst.ImportFrom(
              module=cst.Attribute(value=cst.Name(root_pkg), attr=cst.Name("nn")),
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

  def parse_to_graph(self, code: str | None = None) -> LogicalGraph:
    """Parse the source payload into a normalized LogicalGraph.

    Args:
        code: Optional override for the source code payload.

    Returns:
        LogicalGraph: The parsed and normalized computation graph.
    """
    src = (code or self.code).strip()
    if not src:
      return LogicalGraph()

    if src.startswith(("{", "[")):
      parser = IrJsonParser()
      graph = parser.parse(src)
    else:
      parser_py = IrPythonParser()
      graph = parser_py.parse(src)

    lifter = IrLifter()
    return lifter.lift(graph)
