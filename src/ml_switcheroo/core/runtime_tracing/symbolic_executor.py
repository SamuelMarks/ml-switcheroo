"""Line-Accurate Symbolic Forward CST Executor for On-Device Tracing.

Defines ``_ForwardSymbolicExecutor`` with modular statement, expression, call,
and tensor-creation evaluators (each with McCabe complexity <= 15).
"""

from __future__ import annotations

import typing

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider

from ml_switcheroo.core.runtime_models import (
  BranchResolutionRecord,
  ConstructorDimResolution,
  DeclaredLayerSpec,
  NodeRuntimeTelemetry,
  _extract_int_from_expr,
  _get_call_leaf_name,
)
from ml_switcheroo.core.runtime_tensor import (
  OnDeviceTraceTensor,
  _matmul_2d,
)
from ml_switcheroo.core.runtime_tracing.creation_eval import (
  eval_creation_call,
  eval_inline_ctor_call,
)
from ml_switcheroo.core.runtime_tracing.sandbox import _TracedLayerInstance

_INLINE_CTOR_LEAVES = frozenset({"Linear", "LazyLinear", "LayerNorm", "Dense", "Conv1d", "Conv2d", "Conv3d"})
_CREATION_CALL_LEAVES = frozenset({"zeros", "ones", "full", "empty", "randn", "rand", "arange", "linspace"})

_UNARY_TENSOR_METHODS: dict[str, typing.Callable[[OnDeviceTraceTensor], OnDeviceTraceTensor]] = {
  "relu": lambda t: t.relu(),
  "sigmoid": lambda t: t.sigmoid(),
  "tanh": lambda t: t.tanh(),
  "sum": lambda t: t.sum(),
  "mean": lambda t: t.mean(),
  "transpose": lambda t: t.transpose(),
  "permute": lambda t: t.transpose(),
}

_UNARY_FUNC_OPS: dict[str, typing.Callable[[OnDeviceTraceTensor], OnDeviceTraceTensor]] = {
  "relu": lambda t: t.relu(),
  "gelu": lambda t: t.relu(),
  "silu": lambda t: t.relu(),
  "elu": lambda t: t.relu(),
  "sigmoid": lambda t: t.sigmoid(),
  "tanh": lambda t: t.tanh(),
  "exp": lambda t: t.exp(),
  "flatten": lambda t: t.flatten(1),
  "sum": lambda t: t.sum(),
  "reduce_sum": lambda t: t.sum(),
  "mean": lambda t: t.mean(),
  "reduce_mean": lambda t: t.mean(),
}

_BINARY_FUNC_OPS: dict[str, typing.Callable[[OnDeviceTraceTensor, OnDeviceTraceTensor], OnDeviceTraceTensor]] = {
  "add": lambda a, b: a + b,
  "mul": lambda a, b: a * b,
  "multiply": lambda a, b: a * b,
  "matmul": _matmul_2d,
  "dot": _matmul_2d,
}


class _ForwardSymbolicExecutor(cst.CSTVisitor):
  """Line-accurate CST evaluator that executes forward statements on ``OnDeviceTraceTensor``."""

  METADATA_DEPENDENCIES = (PositionProvider,)

  def __init__(
    self,
    declared_layers: dict[str, DeclaredLayerSpec],
    input_tensors: list[OnDeviceTraceTensor],
    seed: int,
  ) -> None:
    """Initialize the forward symbolic executor.

    Args:
        declared_layers: Map of layer attributes discovered in ``__init__``/``setup``.
        input_tensors: Seeded input ``OnDeviceTraceTensor`` list.
        seed: Random seed for parameter initialization.
    """
    self.declared_layers = declared_layers
    self.input_tensors = input_tensors
    self.seed = seed
    self.env: dict[str, OnDeviceTraceTensor] = {}
    self.parameters: dict[str, OnDeviceTraceTensor] = {}
    self.resolved_constructors: dict[str, ConstructorDimResolution] = {}
    self.resolved_branches: list[BranchResolutionRecord] = []
    self.loop_trip_counts: dict[int, int] = {}
    self.node_telemetry: dict[str, NodeRuntimeTelemetry] = {}
    self.line_telemetry: dict[int, NodeRuntimeTelemetry] = {}
    self.traced_activations: list[tuple[str, int, OnDeviceTraceTensor]] = []
    self.output_tensor: OnDeviceTraceTensor = input_tensors[0]
    self._layer_instances: dict[str, _TracedLayerInstance] = {
      name: _TracedLayerInstance(spec, seed + idx * 17, self.resolved_constructors, self.parameters)
      for idx, (name, spec) in enumerate(declared_layers.items())
    }
    self._node_counter: int = 0

  @staticmethod
  def _locate_entry_function(module: cst.Module) -> cst.FunctionDef | None:
    """Locate the primary module inference method or top-level function in ``module``.

    Args:
        module: Metadata-wrapped ``cst.Module``.

    Returns:
        Target ``cst.FunctionDef`` if present, otherwise ``None``.
    """
    target_fn: cst.FunctionDef | None = None
    for stmt in module.body:
      if isinstance(stmt, cst.ClassDef):
        for body_stmt in stmt.body.body:
          if isinstance(body_stmt, cst.FunctionDef) and body_stmt.name.value in ("forward", "__call__", "call"):
            target_fn = body_stmt
            break
      elif isinstance(stmt, cst.FunctionDef) and target_fn is None:
        target_fn = stmt
    return target_fn

  def execute_module(self, tree: cst.Module) -> OnDeviceTraceTensor:
    """Locate the primary inference function/method in ``tree`` and execute its body.

    Args:
        tree: Parsed LibCST module.

    Returns:
        Final forward output ``OnDeviceTraceTensor``.
    """
    wrapper = MetadataWrapper(tree)
    pos_map = wrapper.resolve(PositionProvider)
    target_fn = self._locate_entry_function(wrapper.module)
    if target_fn is None:
      return self.output_tensor

    arg_idx = 0
    for param in target_fn.params.params:
      pname = param.name.value
      if pname in ("self", "cls", "rngs", "key", "training"):
        continue
      self.env[pname] = self.input_tensors[arg_idx] if arg_idx < len(self.input_tensors) else self.input_tensors[0]
      arg_idx += 1

    if not self.env:
      self.env["x"] = self.input_tensors[0]

    self._exec_suite(target_fn.body.body, pos_map)
    return self.output_tensor

  def _exec_simple_line(self, stmt: cst.SimpleStatementLine, lineno: int) -> bool:
    """Execute small statements within ``stmt``, returning True if ``return`` was executed.

    Args:
        stmt: ``cst.SimpleStatementLine`` node.
        lineno: 1-based source line number.

    Returns:
        ``True`` if execution hit a ``return`` statement.
    """
    for small in stmt.body:
      if isinstance(small, cst.Assign) and small.targets:
        var_name = self._target_name(small.targets[0].target)
        in_shapes = self._collect_expr_input_shapes(small.value)
        val, op_label = self._eval_expr(small.value)
        self.env[var_name] = val
        self.output_tensor = val
        self._record_telemetry(var_name, op_label, lineno, in_shapes, val)
      elif isinstance(small, cst.AugAssign):
        var_name = self._target_name(small.target)
        lhs = self.env.get(var_name, self.output_tensor)
        rhs, op_label = self._eval_expr(small.value)
        val = lhs + rhs
        self.env[var_name] = val
        self.output_tensor = val
        self._record_telemetry(var_name, f"AugAssign({op_label})", lineno, [list(lhs.shape)], val)
      elif isinstance(small, cst.Return) and small.value is not None:
        in_shapes = self._collect_expr_input_shapes(small.value)
        val, op_label = self._eval_expr(small.value)
        self.output_tensor = val
        self._record_telemetry("return", op_label, lineno, in_shapes, val)
        return True
      elif isinstance(small, cst.Expr):
        val, _ = self._eval_expr(small.value)
        self.output_tensor = val
    return False

  def _exec_if_stmt(
    self,
    stmt: cst.If,
    lineno: int,
    pos_map: typing.Mapping[cst.CSTNode, CodeRange],
  ) -> bool:
    """Evaluate ``stmt`` condition, record branch telemetry, and execute the taken arm.

    Args:
        stmt: ``cst.If`` compound statement.
        lineno: 1-based source line number.
        pos_map: ``PositionProvider`` mapping.

    Returns:
        ``True`` if the executed branch returned early.
    """
    cond_src = cst.Module([]).code_for_node(stmt.test).strip()
    cond_tensor, _ = self._eval_expr(stmt.test)
    taken = bool(cond_tensor)
    self.resolved_branches.append(
      BranchResolutionRecord(
        lineno=lineno or 0,
        condition_src=cond_src,
        taken_branch="true" if taken else "false",
      )
    )
    if taken:
      return self._exec_suite(stmt.body.body, pos_map)
    if stmt.orelse is None:
      return False
    else_suite = stmt.orelse.body.body if isinstance(stmt.orelse, cst.Else) else [stmt.orelse]
    return self._exec_suite(else_suite, pos_map)

  def _exec_while_stmt(
    self,
    stmt: cst.While,
    lineno: int,
    pos_map: typing.Mapping[cst.CSTNode, CodeRange],
  ) -> bool:
    """Execute ``stmt`` up to 3 bounded iterations while recording ``loop_trip_counts``.

    Args:
        stmt: ``cst.While`` compound statement.
        lineno: 1-based source line number.
        pos_map: ``PositionProvider`` mapping.

    Returns:
        ``True`` if the loop body returned early.
    """
    trips = 0
    while trips < 3:
      cond_tensor, _ = self._eval_expr(stmt.test)
      if not bool(cond_tensor):
        break
      trips += 1
      if self._exec_suite(stmt.body.body, pos_map):
        self.loop_trip_counts[lineno] = trips
        return True
    self.loop_trip_counts[lineno] = trips
    return False

  def _exec_for_stmt(
    self,
    stmt: cst.For,
    lineno: int,
    pos_map: typing.Mapping[cst.CSTNode, CodeRange],
  ) -> bool:
    """Execute ``stmt`` over its resolved range bound while recording ``loop_trip_counts``.

    Args:
        stmt: ``cst.For`` compound statement.
        lineno: 1-based source line number.
        pos_map: ``PositionProvider`` mapping.

    Returns:
        ``True`` if the loop body returned early.
    """
    trips = self._eval_for_trip_count(stmt.iter)
    self.loop_trip_counts[lineno] = trips
    loop_var = stmt.target.value if isinstance(stmt.target, cst.Name) else None
    for iter_idx in range(min(trips, 16)):
      if loop_var is not None:
        self.env[loop_var] = OnDeviceTraceTensor(shape=(), data=[float(iter_idx)], dtype="int32", requires_grad=False)
      if self._exec_suite(stmt.body.body, pos_map):
        return True
    return False

  def _exec_suite(
    self,
    statements: typing.Sequence[cst.BaseStatement | cst.BaseSmallStatement],
    pos_map: typing.Mapping[cst.CSTNode, CodeRange],
  ) -> bool:
    """Execute a sequence of CST statements, returning True if a ``return`` was hit.

    Args:
        statements: Sequence of CST statements.
        pos_map: LibCST ``PositionProvider`` mapping.

    Returns:
        True if execution terminated via a ``return`` statement.
    """
    for stmt in statements:
      lineno = self._lineno(stmt, pos_map)
      if isinstance(stmt, cst.SimpleStatementLine):
        if self._exec_simple_line(stmt, lineno):
          return True
      elif isinstance(stmt, cst.If):
        if self._exec_if_stmt(stmt, lineno, pos_map):
          return True
      elif isinstance(stmt, cst.While):
        if self._exec_while_stmt(stmt, lineno, pos_map):
          return True
      elif isinstance(stmt, cst.For) and self._exec_for_stmt(stmt, lineno, pos_map):
        return True
    return False

  def _eval_for_trip_count(self, iter_expr: cst.BaseExpression) -> int:
    """Evaluate the integer trip count of a ``for ... in range(...)`` iterator expression.

    Args:
        iter_expr: Iterator CST expression on ``cst.For``.

    Returns:
        Non-negative integer trip count (defaults to 1 if unresolvable).
    """
    if isinstance(iter_expr, cst.Call) and _get_call_leaf_name(iter_expr.func) == "range" and iter_expr.args:
      if len(iter_expr.args) == 1:
        stop_val = self._eval_dim_expr(iter_expr.args[0].value)
        return max(0, stop_val) if stop_val is not None else 1
      start_val = self._eval_dim_expr(iter_expr.args[0].value) or 0
      stop_val = self._eval_dim_expr(iter_expr.args[1].value)
      step_val = self._eval_dim_expr(iter_expr.args[2].value) if len(iter_expr.args) >= 3 else 1
      if stop_val is not None and step_val is not None and step_val != 0:
        return max(0, (stop_val - start_val + (step_val - (1 if step_val > 0 else -1))) // step_val)
    return 1

  def _eval_binary_op(self, expr: cst.BinaryOperation) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate a ``cst.BinaryOperation`` node on traced tensors.

    Args:
        expr: Binary operation CST node.

    Returns:
        Tuple of ``(result_tensor, op_name)``.
    """
    lhs, _ = self._eval_expr(expr.left)
    rhs, _ = self._eval_expr(expr.right)
    if isinstance(expr.operator, cst.Add):
      return lhs + rhs, "Add"
    if isinstance(expr.operator, cst.Subtract):
      return lhs - rhs, "Sub"
    if isinstance(expr.operator, cst.Multiply):
      return lhs * rhs, "Mul"
    if isinstance(expr.operator, cst.Divide):
      return lhs / rhs, "Div"
    if isinstance(expr.operator, cst.MatrixMultiply):
      return _matmul_2d(lhs, rhs), "MatMul"
    return lhs + rhs, "BinaryOp"

  def _eval_comparison_op(self, expr: cst.Comparison) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate a ``cst.Comparison`` node into a boolean mask ``OnDeviceTraceTensor``.

    Args:
        expr: Comparison CST node.

    Returns:
        Tuple of ``(mask_tensor, op_name)``.
    """
    lhs, _ = self._eval_expr(expr.left)
    comp = expr.comparisons[0]
    rhs, _ = self._eval_expr(comp.comparator)
    if isinstance(comp.operator, (cst.GreaterThan, cst.GreaterThanEqual)):
      return lhs > rhs, "Greater"
    if isinstance(comp.operator, (cst.LessThan, cst.LessThanEqual)):
      return lhs < rhs, "Less"
    return lhs > 0.0, "Compare"

  def _eval_subscript_expr(self, expr: cst.Subscript) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate ``x.shape[idx]``, boolean mask indexing, or slice indexing on ``expr``.

    Args:
        expr: Subscript CST node.

    Returns:
        Tuple of ``(indexed_tensor, op_name)``.
    """
    if (
      isinstance(expr.value, cst.Attribute)
      and expr.value.attr.value == "shape"
      and expr.slice
      and isinstance(expr.slice[0].slice, cst.Index)
    ):
      base_t, _ = self._eval_expr(expr.value.value)
      idx_int = _extract_int_from_expr(expr.slice[0].slice.value)
      if idx_int is not None and -len(base_t.shape) <= idx_int < len(base_t.shape):
        dim_val = base_t.shape[idx_int]
        return OnDeviceTraceTensor(shape=(), data=[float(dim_val)], dtype="int32", requires_grad=False), "ShapeDim"

    base, _ = self._eval_expr(expr.value)
    first_slice = expr.slice[0].slice
    if isinstance(first_slice, cst.Index):
      idx_val, _ = self._eval_expr(first_slice.value)
      if idx_val.dtype == "bool":
        return base[idx_val], "BooleanMaskIndex"
    return base, "Slice"

  def _eval_expr(self, expr: cst.BaseExpression) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate a CST expression to an ``(OnDeviceTraceTensor, op_type)`` pair.

    Args:
        expr: LibCST expression node.

    Returns:
        Tuple of ``(evaluated_tensor, operation_name)``.
    """
    if isinstance(expr, cst.Name):
      return self.env.get(expr.value, self.output_tensor), "Identity"
    if isinstance(expr, cst.Integer):
      return OnDeviceTraceTensor(shape=(), data=[float(int(expr.value))], requires_grad=False), "Constant"
    if isinstance(expr, cst.Float):
      return OnDeviceTraceTensor(shape=(), data=[float(expr.value)], requires_grad=False), "Constant"
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Minus):
      inner, _ = self._eval_expr(expr.expression)
      return inner * -1.0, "Neg"
    if isinstance(expr, cst.BinaryOperation):
      return self._eval_binary_op(expr)
    if isinstance(expr, cst.Comparison):
      return self._eval_comparison_op(expr)
    if isinstance(expr, cst.Subscript):
      return self._eval_subscript_expr(expr)
    if isinstance(expr, cst.Call):
      return self._eval_call(expr)
    return self.output_tensor, "Expr"

  def _eval_inline_ctor_call(
    self, call: cst.Call, ctor_call: cst.Call, ctor_leaf: str
  ) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate an inline constructor invocation such as ``nn.Linear(x.shape[-1], 10)(x)``.

    Args:
        call: Outer invocation ``cst.Call`` node.
        ctor_call: Inner constructor ``cst.Call`` node.
        ctor_leaf: Constructor leaf name.

    Returns:
        Tuple of ``(evaluated_tensor, op_type)``.
    """
    return eval_inline_ctor_call(self, call, ctor_call, ctor_leaf)

  def _try_eval_tensor_method_call(
    self, call: cst.Call, attr_func: cst.Attribute
  ) -> tuple[OnDeviceTraceTensor, str] | None:
    """Evaluate tensor instance methods (``size``, ``flatten``, ``reshape``, ``view``, etc.) if receiver is in ``env``.

    Args:
        call: ``cst.Call`` node.
        attr_func: ``cst.Attribute`` callable target.

    Returns:
        Evaluated ``(tensor, op_name)`` if handled as a tensor method, otherwise ``None``.
    """
    root_node: cst.BaseExpression = attr_func.value
    while isinstance(root_node, cst.Attribute):
      root_node = root_node.value
    if isinstance(root_node, cst.Name) and root_node.value not in self.env:
      return None

    recv, _ = self._eval_expr(attr_func.value)
    method_name = attr_func.attr.value
    if method_name == "size":
      idx_int = _extract_int_from_expr(call.args[0].value) if call.args else 0
      dim_val = recv.shape[idx_int] if (idx_int is not None and -len(recv.shape) <= idx_int < len(recv.shape)) else 1
      return OnDeviceTraceTensor(shape=(), data=[float(dim_val)], dtype="int32", requires_grad=False), "SizeDim"
    if method_name == "flatten":
      parsed = _extract_int_from_expr(call.args[0].value) if call.args else None
      return recv.flatten(parsed if parsed is not None else 1), "flatten"
    if method_name in ("reshape", "view"):
      return recv.reshape(*self._parse_call_int_args(call.args)), method_name
    if method_name in _UNARY_TENSOR_METHODS:
      return _UNARY_TENSOR_METHODS[method_name](recv), method_name
    return None

  def _eval_call(self, call: cst.Call) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate a function, method, or ``self.<layer>(...)`` call expression.

    Args:
        call: LibCST Call expression.

    Returns:
        Tuple of ``(evaluated_tensor, operation_name)``.
    """
    if isinstance(call.func, cst.Attribute) and isinstance(call.func.value, cst.Name) and call.func.value.value == "self":
      layer_attr = call.func.attr.value
      arg_tensor = self._eval_expr(call.args[0].value)[0] if call.args else self.output_tensor
      if layer_attr not in self._layer_instances:
        spec = DeclaredLayerSpec(layer_attr, "Linear", None, arg_tensor.shape[-1] if arg_tensor.shape else 4, True)
        self._layer_instances[layer_attr] = _TracedLayerInstance(
          spec, self.seed + 99, self.resolved_constructors, self.parameters
        )
      inst = self._layer_instances[layer_attr]
      return inst(arg_tensor), inst.spec.op_type

    if isinstance(call.func, cst.Call):
      ctor_leaf = _get_call_leaf_name(call.func.func)
      if ctor_leaf in _INLINE_CTOR_LEAVES:
        return self._eval_inline_ctor_call(call, call.func, ctor_leaf)

    if isinstance(call.func, cst.Attribute):
      method_res = self._try_eval_tensor_method_call(call, call.func)
      if method_res is not None:
        return method_res

    leaf = _get_call_leaf_name(call.func)
    lower = leaf.lower()
    if lower in _CREATION_CALL_LEAVES:
      return self._eval_creation_call(lower, leaf, call)

    arg0 = self._eval_expr(call.args[0].value)[0] if call.args else self.output_tensor
    if lower in _UNARY_FUNC_OPS:
      return _UNARY_FUNC_OPS[lower](arg0), leaf
    if lower == "reshape":
      return arg0.reshape(*self._parse_call_int_args(call.args[1:])), leaf
    if lower in _BINARY_FUNC_OPS:
      arg1 = self._eval_expr(call.args[1].value)[0] if len(call.args) > 1 else arg0
      return _BINARY_FUNC_OPS[lower](arg0, arg1), leaf
    if lower in ("nonzero", "unique"):
      masked = arg0[arg0 > 0.0]
      masked.dynamic_reason = f"Data-dependent {leaf} output shape resolved on-device"
      return masked, leaf
    return arg0, leaf or "Call"

  def _eval_creation_call(
    self,
    lower: str,
    leaf: str,
    call: cst.Call,
  ) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate a tensor creation call (`zeros`, `ones`, `full`, `empty`, `randn`, `rand`, `arange`, `linspace`).

    Args:
        lower: Lowercase leaf name of the creation function.
        leaf: Original leaf name of the creation function.
        call: LibCST Call expression.

    Returns:
        Tuple of ``(created_tensor, operation_name)``.
    """
    return eval_creation_call(self, lower, leaf, call)

  def _eval_dim_expr(self, expr: cst.BaseExpression) -> int | None:
    """Evaluate a dimension expression to an ``int``, or ``None`` if unresolvable.

    Args:
        expr: LibCST dimension expression node.

    Returns:
        Evaluated integer dimension, or ``None``.
    """
    lit = _extract_int_from_expr(expr)
    if lit is not None:
      return lit
    val_t, _ = self._eval_expr(expr)
    if val_t.shape == () and val_t.data:
      return round(val_t.data[0])
    return None

  def _parse_call_int_args(self, args: typing.Sequence[cst.Arg]) -> list[int]:
    """Extract integer arguments from call arguments, expanding tuple/list literals.

    Args:
        args: Sequence of LibCST ``Arg`` nodes.

    Returns:
        List of parsed integer dimensions.
    """
    dims: list[int] = []
    for arg in args:
      val = arg.value
      candidates = [el.value for el in val.elements] if isinstance(val, (cst.Tuple, cst.List)) else [val]
      for cand in candidates:
        d = self._eval_dim_expr(cand)
        if d is not None:
          dims.append(d)
    return dims

  def _collect_expr_input_shapes(self, expr: cst.BaseExpression) -> list[list[int]]:
    """Collect shapes of known variable references appearing inside ``expr``.

    Args:
        expr: LibCST expression node.

    Returns:
        List of input shape lists.
    """

    class _NameCollector(cst.CSTVisitor):
      """Visitor collecting variable names referenced in an expression."""

      def __init__(self, env: dict[str, OnDeviceTraceTensor]) -> None:
        """Initialize with current variable environment.

        Args:
            env: Current variable map.
        """
        self.env = env
        self.found: list[list[int]] = []

      def visit_Name(self, node: cst.Name) -> None:
        """Record tensor shape if ``node.value`` is in ``self.env``.

        Args:
            node: Name node.
        """
        if node.value in self.env:
          s = list(self.env[node.value].shape)
          if s not in self.found:
            self.found.append(s)

    collector = _NameCollector(self.env)
    expr.visit(collector)
    return list(collector.found)

  def _record_telemetry(
    self,
    var_name: str,
    op_type: str,
    lineno: int,
    input_shapes: list[list[int]],
    tensor: OnDeviceTraceTensor,
  ) -> None:
    """Create a ``NodeRuntimeTelemetry`` entry and register it by node ID and line number.

    Args:
        var_name: Variable name or ``'return'``.
        op_type: Operation or layer type name.
        lineno: 1-based source line number.
        input_shapes: Input tensor shapes to this statement.
        tensor: Resulting ``OnDeviceTraceTensor``.
    """
    node_id = f"node_{self._node_counter}_{var_name}"
    self._node_counter += 1
    tel = NodeRuntimeTelemetry(
      node_id=node_id,
      var_name=var_name,
      op_type=op_type,
      lineno=lineno,
      input_shapes=input_shapes,
      output_shape=list(tensor.shape),
      dtype=tensor.dtype,
      is_dynamic=tensor.is_dynamic,
      dynamic_reason=tensor.dynamic_reason,
    )
    self.node_telemetry[node_id] = tel
    self.line_telemetry[lineno] = tel
    self.traced_activations.append((node_id, lineno, tensor))

  @staticmethod
  def _target_name(node: cst.BaseExpression) -> str:
    """Extract variable name from an assignment target expression.

    Args:
        node: Assignment target CST expression.

    Returns:
        Target variable identifier string.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      return node.attr.value
    return "tmp"

  @staticmethod
  def _lineno(
    node: cst.CSTNode,
    pos_map: typing.Mapping[cst.CSTNode, CodeRange],
  ) -> int:
    """Resolve the 1-based start line number of a CST node from ``PositionProvider``.

    Args:
        node: CST node to look up.
        pos_map: PositionProvider metadata map.

    Returns:
        1-based line number (0 if not present).
    """
    pos = pos_map.get(node)
    return int(pos.start.line) if pos is not None else 0


__all__ = ["_ForwardSymbolicExecutor"]
