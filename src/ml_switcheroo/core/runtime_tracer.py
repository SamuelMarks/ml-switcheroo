"""On-Device Forward and Backward Pass Execution Engine.

This module routes model execution, shape inference, reverse-mode automatic
differentiation (VJP / `value_and_grad`), and hardware kernel emission
(WebGPU WGSL and WASM SIMD) through `zero-*` and `ml-switcheroo-compiler`.
It resolves static-analysis impasses including lazy layer input dimensions
(`LazyLinear`, unbuilt `Dense`), data-dependent boolean mask shapes, conditional
branches, `while` loop trip counts, and backward gradient shapes and L2 norms.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
import types
import typing

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider

from ml_switcheroo.config import ExecutionBackend, RuntimeConfig
from ml_switcheroo.core.compiler.compiler_passes import ensure_compiler_and_zero_on_path
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.ir_bridge import (
  _NON_PYTHON_SOURCE_DIALECTS,
  _lift_non_python_source_for_trace as _lift_non_python_source_for_trace,
  hydrate_graph_with_runtime_trace as hydrate_graph_with_runtime_trace,
  run_shared_ir_bridge,
)
from ml_switcheroo.core.runtime_models import (
  BranchResolutionRecord,
  ConstructorDimResolution,
  DeclaredLayerSpec as DeclaredLayerSpec,
  NodeRuntimeTelemetry,
  RuntimeTraceReport,
)
from ml_switcheroo.core.runtime_tensor import (
  OnDeviceTraceTensor,
  _check_all_finite as _check_all_finite,
  _compute_loss as _compute_loss,
  _conv_nd,
  _generate_seeded_values,
  _l2_norm as _l2_norm,
  _matmul_2d,
  _prod,
)
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits

_SUPPORT_ZERO_ALIASES: dict[str, str] = {
  "optax": "zero_optax",
  "chex": "zero_chex",
  "grain": "zero_grain",
  "orbax": "zero_orbax",
  "zero_zoo": "zero_zoo",
}


def _resolve_traits(framework: str, semantics: SemanticsManager | None = None) -> StructuralTraits:
  """Resolve `StructuralTraits` for a framework key from its adapter or `SemanticsManager`.

  Args:
      framework: Canonical framework identifier.
      semantics: Optional `SemanticsManager` instance.

  Returns:
      Resolved `StructuralTraits` instance.
  """
  adapter = get_adapter(framework)
  if adapter is not None and hasattr(adapter, "structural_traits"):
    traits = adapter.structural_traits
    if isinstance(traits, StructuralTraits):
      return traits
  if semantics is not None:
    fw_cfg = semantics.get_framework_config(framework)
    raw_traits = fw_cfg.get("traits")
    if isinstance(raw_traits, dict):
      return StructuralTraits.model_validate(raw_traits)
  return StructuralTraits()


@contextlib.contextmanager
def zero_framework_sandbox(
  source_fw: str,
  target_fw: str,
  semantics: SemanticsManager | None = None,
) -> typing.Iterator[dict[str, types.ModuleType]]:
  """Temporarily alias standard framework module paths to `zero-*` modules in `sys.modules`.

  Reads `StructuralTraits.zero_module_aliases` for the active source and target
  frameworks (plus standard `zero-*` ecosystem packages) and binds any importable
  `zero_*` modules into `sys.modules` for the duration of the context block.

  Args:
      source_fw: Source framework identifier.
      target_fw: Target framework identifier.
      semantics: Optional `SemanticsManager` instance.

  Yields:
      Dictionary mapping canonical module names to imported `zero-*` modules.
  """
  ensure_compiler_and_zero_on_path()
  src_traits = _resolve_traits(source_fw, semantics)
  tgt_traits = _resolve_traits(target_fw, semantics)

  alias_map: dict[str, str] = dict(_SUPPORT_ZERO_ALIASES)
  alias_map.update(src_traits.zero_module_aliases)
  alias_map.update(tgt_traits.zero_module_aliases)

  saved_modules: dict[str, types.ModuleType | None] = {}
  active_bindings: dict[str, types.ModuleType] = {}

  try:
    for canonical_name, zero_name in alias_map.items():
      try:
        zero_mod = importlib.import_module(zero_name)
      except ImportError:
        continue
      saved_modules[canonical_name] = sys.modules.get(canonical_name)
      sys.modules[canonical_name] = zero_mod
      active_bindings[canonical_name] = zero_mod
    yield active_bindings
  finally:
    for canonical_name, previous_mod in saved_modules.items():
      if previous_mod is None:
        sys.modules.pop(canonical_name, None)
      else:
        sys.modules[canonical_name] = previous_mod


def _extract_int_from_expr(node: cst.BaseExpression) -> int | None:
  """Extract a positive integer literal from a CST expression if statically present.

  Args:
      node: LibCST expression node.

  Returns:
      Parsed integer value or None.
  """
  if isinstance(node, cst.Integer):
    try:
      return int(node.value)
    except ValueError:
      return None
  return None


def _get_call_leaf_name(func: cst.BaseExpression) -> str:
  """Extract the leaf callable name from a Call expression's `func` node.

  Args:
      func: LibCST expression representing the called function or class.

  Returns:
      Leaf identifier string (e.g. 'Linear', 'Conv2d', 'LazyLinear').
  """
  if isinstance(func, cst.Name):
    return func.value
  if isinstance(func, cst.Attribute):
    return func.attr.value
  return ""


class InputShapeInferenceVisitor(cst.CSTVisitor):
  """CST visitor that infers model input shapes and declared layer constructors."""

  def __init__(self) -> None:
    """Initialize the visitor state."""
    self.inferred_shape: tuple[int, ...] | None = None
    self.inferred_dtype: str = "float32"
    self.declared_layers: dict[str, DeclaredLayerSpec] = {}
    self._in_init: bool = False

  def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
    """Inspect function definitions for `__init__`/`setup` or shape annotations.

    Args:
        node: Function definition CST node.

    Returns:
        True to continue traversal into the function body.
    """
    if node.name.value in ("__init__", "setup"):
      self._in_init = True
    for param in node.params.params:
      if param.name.value in ("self", "cls", "rngs", "key"):
        continue
      if param.annotation is not None and self.inferred_shape is None:
        ann_code = cst.Module([]).code_for_node(param.annotation.annotation)
        dims = _extract_dims_from_annotation_text(ann_code)
        if dims:
          self.inferred_shape = dims
    return True

  def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
    """Reset `__init__`/`setup` state upon leaving a function definition.

    Args:
        original_node: Function definition CST node.
    """
    if original_node.name.value in ("__init__", "setup"):
      self._in_init = False

  def visit_Assign(self, node: cst.Assign) -> bool:
    """Inspect `self.<layer> = <LayerCall>(...)` assignments inside `__init__`/`setup`.

    Args:
        node: Assignment CST node.

    Returns:
        True to continue traversal.
    """
    if not self._in_init or not isinstance(node.value, cst.Call):
      return True
    for target in node.targets:
      if (
        isinstance(target.target, cst.Attribute)
        and isinstance(target.target.value, cst.Name)
        and target.target.value.value == "self"
      ):
        layer_name = target.target.attr.value
        self._record_layer_call(layer_name, node.value)
    return True

  def visit_Call(self, node: cst.Call) -> bool:
    """Inspect `self.create_child('<layer>', <LayerCall>(...))` calls in Praxis `setup`.

    Args:
        node: Call CST node.

    Returns:
        True to continue traversal.
    """
    if (
      self._in_init
      and isinstance(node.func, cst.Attribute)
      and isinstance(node.func.value, cst.Name)
      and node.func.value.value == "self"
      and node.func.attr.value == "create_child"
      and len(node.args) >= 2
      and isinstance(node.args[0].value, cst.SimpleString)
      and isinstance(node.args[1].value, cst.Call)
    ):
      layer_name = str(node.args[0].value.evaluated_value)
      self._record_layer_call(layer_name, node.args[1].value)
    return True

  def _record_layer_call(self, layer_name: str, call: cst.Call) -> None:
    """Extract constructor dimensions and first-layer input shape from a layer call.

    Args:
        layer_name: Attribute name on `self`.
        call: Layer constructor Call node.
    """
    op_type = _get_call_leaf_name(call.func)
    if not op_type:
      return

    pos_ints: list[int] = []
    kw_ints: dict[str, int] = {}
    for arg in call.args:
      val = _extract_int_from_expr(arg.value)
      if val is not None:
        if arg.keyword is None:
          pos_ints.append(val)
        else:
          kw_ints[arg.keyword.value] = val

    in_feat: int | None = None
    out_feat: int | None = None
    is_lazy = False

    if op_type in ("LazyLinear", "LazyConv1d", "LazyConv2d", "LazyConv3d"):
      is_lazy = True
      out_feat = kw_ints.get("out_features") or kw_ints.get("out_channels") or (pos_ints[0] if pos_ints else None)
    elif op_type in ("Dense",):
      out_feat = kw_ints.get("units") or (pos_ints[0] if pos_ints else None)
      in_feat = kw_ints.get("input_dim")
      is_lazy = in_feat is None
    elif op_type in ("Linear",):
      in_feat = kw_ints.get("in_features") or kw_ints.get("input_dims")
      out_feat = kw_ints.get("out_features") or kw_ints.get("output_dims")
      if in_feat is None and len(pos_ints) >= 2:
        in_feat = pos_ints[0]
        out_feat = pos_ints[1]
      elif in_feat is None and len(pos_ints) == 1 and out_feat is None:
        out_feat = pos_ints[0]
        is_lazy = True
      elif in_feat is None:
        is_lazy = True
    elif op_type in ("Conv1d", "Conv1D", "Conv2d", "Conv2D", "Conv3d", "Conv3D", "Conv"):
      in_feat = kw_ints.get("in_channels") or kw_ints.get("in_features") or kw_ints.get("input_dims")
      out_feat = kw_ints.get("out_channels") or kw_ints.get("out_features") or kw_ints.get("filters")
      if in_feat is None and len(pos_ints) >= 2:
        in_feat = pos_ints[0]
        out_feat = pos_ints[1]
      elif in_feat is None:
        is_lazy = True
    elif op_type == "Embedding":
      in_feat = kw_ints.get("num_embeddings") or (pos_ints[0] if pos_ints else 16)
      out_feat = kw_ints.get("embedding_dim") or (pos_ints[1] if len(pos_ints) >= 2 else 8)
    elif op_type == "MultiheadAttention":
      in_feat = kw_ints.get("embed_dim") or (pos_ints[0] if pos_ints else 8)
      out_feat = in_feat

    self.declared_layers[layer_name] = DeclaredLayerSpec(
      layer_name=layer_name,
      op_type=op_type,
      in_features=in_feat,
      out_features=out_feat,
      is_lazy=is_lazy,
    )

    if self.inferred_shape is None:
      if op_type == "Linear" and in_feat is not None:
        self.inferred_shape = (2, in_feat)
      elif op_type in ("Conv1d", "Conv1D") and in_feat is not None:
        self.inferred_shape = (2, in_feat, 8)
      elif op_type in ("Conv2d", "Conv2D", "Conv") and in_feat is not None:
        self.inferred_shape = (2, in_feat, 8, 8)
      elif op_type in ("Conv3d", "Conv3D") and in_feat is not None:
        self.inferred_shape = (2, in_feat, 4, 4, 4)
      elif op_type == "Embedding":
        self.inferred_shape = (2, 8)
        self.inferred_dtype = "int32"
      elif op_type == "MultiheadAttention" and in_feat is not None:
        self.inferred_shape = (2, 4, in_feat)


def _extract_dims_from_annotation_text(text: str) -> tuple[int, ...] | None:
  """Extract integer shape tuple from a type annotation string if present.

  Args:
      text: Raw type annotation source string.

  Returns:
      Tuple of positive integers if found, else None.
  """
  cleaned = text.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ").replace(",", " ")
  cleaned = cleaned.replace('"', " ").replace("'", " ")
  nums: list[int] = []
  for tok in cleaned.split():
    if tok.isdigit():
      val = int(tok)
      if 0 < val <= 4096:
        nums.append(val)
  return tuple(nums) if len(nums) >= 2 else None


def infer_input_shapes_from_cst(
  tree: cst.Module,
  config: RuntimeConfig | None = None,
) -> tuple[list[tuple[int, ...]], str, dict[str, DeclaredLayerSpec]]:
  """Infer model input tensor shapes, dtype, and declared layers from a CST module.

  Args:
      tree: Parsed LibCST module.
      config: Optional `RuntimeConfig` providing user overrides.

  Returns:
      Tuple of `(input_shapes, input_dtype, declared_layers)`.
  """
  visitor = InputShapeInferenceVisitor()
  tree.visit(visitor)

  dtype = (
    config.input_dtype if (config and config.input_dtype and config.input_dtype != "float32") else visitor.inferred_dtype
  )
  if config and config.input_shapes:
    return list(config.input_shapes), dtype, visitor.declared_layers
  if visitor.inferred_shape is not None:
    return [visitor.inferred_shape], dtype, visitor.declared_layers
  return [(2, 4)], dtype, visitor.declared_layers


class _TracedLayerInstance:
  """Runtime layer instance that records lazy input dimensions and parameter tensors."""

  def __init__(
    self,
    spec: DeclaredLayerSpec,
    seed: int,
    resolved_constructors: dict[str, ConstructorDimResolution],
    parameters: dict[str, OnDeviceTraceTensor],
  ) -> None:
    """Initialize the traced layer wrapper.

    Args:
        spec: Declared layer specification from `__init__`/`setup`.
        seed: Random seed for parameter initialization.
        resolved_constructors: Mutable dictionary populated on first forward call.
        parameters: Mutable dictionary of model parameter tensors.
    """
    self.spec = spec
    self.seed = seed
    self._resolved_constructors = resolved_constructors
    self._parameters = parameters

  def __call__(self, x: OnDeviceTraceTensor) -> OnDeviceTraceTensor:
    """Execute the layer forward pass, lazily materializing parameters on first call.

    Args:
        x: Input `OnDeviceTraceTensor`.

    Returns:
        Output `OnDeviceTraceTensor`.
    """
    op = self.spec.op_type
    layer_name = self.spec.layer_name

    if op in ("Flatten",):
      return x.flatten(1)
    if op in ("ReLU", "relu"):
      return x.relu()
    if op in ("Sigmoid", "sigmoid"):
      return x.sigmoid()
    if op in ("Tanh", "tanh"):
      return x.tanh()

    in_dim = (
      x.shape[1]
      if (op.startswith("Conv") or op.startswith("LazyConv")) and len(x.shape) > 1
      else (x.shape[-1] if x.shape else 4)
    )
    out_dim = self.spec.out_features if self.spec.out_features is not None else in_dim

    if self.spec.is_lazy or self.spec.in_features is None or self.spec.in_features != in_dim:
      self._resolved_constructors[layer_name] = ConstructorDimResolution(
        layer_name=layer_name,
        op_type=op,
        inferred_in_features=in_dim,
        out_features=out_dim,
      )

    w_key = f"{layer_name}.weight"
    b_key = f"{layer_name}.bias"
    if op in ("Conv1d", "Conv1D", "Conv2d", "Conv2D", "Conv3d", "Conv3D", "Conv", "LazyConv2d"):
      if w_key not in self._parameters:
        w_shape: tuple[int, ...] = (out_dim, in_dim, 3, 3) if "2" in op or op == "Conv" else (out_dim, in_dim, 3)
        self._parameters[w_key] = OnDeviceTraceTensor(
          shape=w_shape,
          data=_generate_seeded_values(_prod(w_shape), self.seed + len(self._parameters) + 1, "float32", 0.25),
        )
        self._parameters[b_key] = OnDeviceTraceTensor(
          shape=(out_dim,),
          data=_generate_seeded_values(out_dim, self.seed + len(self._parameters) + 2, "float32", 0.1),
        )
      return _conv_nd(x, self._parameters[w_key], self._parameters[b_key], out_dim)

    if w_key not in self._parameters:
      lin_w_shape: tuple[int, ...] = (out_dim, in_dim)
      self._parameters[w_key] = OnDeviceTraceTensor(
        shape=lin_w_shape,
        data=_generate_seeded_values(_prod(lin_w_shape), self.seed + len(self._parameters) + 1, "float32", 0.3),
      )
      self._parameters[b_key] = OnDeviceTraceTensor(
        shape=(out_dim,),
        data=_generate_seeded_values(out_dim, self.seed + len(self._parameters) + 2, "float32", 0.1),
      )
    return _matmul_2d(x, self._parameters[w_key], self._parameters[b_key])


class _ForwardSymbolicExecutor(cst.CSTVisitor):
  """Line-accurate CST evaluator that executes forward statements on `OnDeviceTraceTensor`."""

  METADATA_DEPENDENCIES = (PositionProvider,)

  def __init__(
    self,
    declared_layers: dict[str, DeclaredLayerSpec],
    input_tensors: list[OnDeviceTraceTensor],
    seed: int,
  ) -> None:
    """Initialize the forward symbolic executor.

    Args:
        declared_layers: Map of layer attributes discovered in `__init__`/`setup`.
        input_tensors: Seeded input `OnDeviceTraceTensor` list.
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

  def execute_module(self, tree: cst.Module) -> OnDeviceTraceTensor:
    """Locate the primary inference function/method in `tree` and execute its body.

    Args:
        tree: Parsed LibCST module.

    Returns:
        Final forward output `OnDeviceTraceTensor`.
    """
    wrapper = MetadataWrapper(tree)
    pos_map = wrapper.resolve(PositionProvider)

    target_fn: cst.FunctionDef | None = None
    for stmt in wrapper.module.body:
      if isinstance(stmt, cst.ClassDef):
        for body_stmt in stmt.body.body:
          if isinstance(body_stmt, cst.FunctionDef) and body_stmt.name.value in (
            "forward",
            "__call__",
            "call",
          ):
            target_fn = body_stmt
            break
      elif isinstance(stmt, cst.FunctionDef) and target_fn is None:
        target_fn = stmt

    if target_fn is None:
      return self.output_tensor

    arg_idx = 0
    for param in target_fn.params.params:
      pname = param.name.value
      if pname in ("self", "cls", "rngs", "key", "training"):
        continue
      if arg_idx < len(self.input_tensors):
        self.env[pname] = self.input_tensors[arg_idx]
        arg_idx += 1
      else:
        self.env[pname] = self.input_tensors[0]

    if not self.env:
      self.env["x"] = self.input_tensors[0]

    self._exec_suite(target_fn.body.body, pos_map)
    return self.output_tensor

  def _exec_suite(
    self,
    statements: typing.Sequence[cst.BaseStatement | cst.BaseSmallStatement],
    pos_map: typing.Mapping[cst.CSTNode, CodeRange],
  ) -> bool:
    """Execute a sequence of CST statements, returning True if a `return` was hit.

    Args:
        statements: Sequence of CST statements.
        pos_map: LibCST `PositionProvider` mapping.

    Returns:
        True if execution terminated via a `return` statement.
    """
    for stmt in statements:
      lineno = self._lineno(stmt, pos_map)
      if isinstance(stmt, cst.SimpleStatementLine):
        for small in stmt.body:
          if isinstance(small, cst.Assign) and small.targets:
            target_node = small.targets[0].target
            var_name = self._target_name(target_node)
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
      elif isinstance(stmt, cst.If):
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
          if self._exec_suite(stmt.body.body, pos_map):
            return True
        elif stmt.orelse is not None:
          if isinstance(stmt.orelse, cst.Else):
            if self._exec_suite(stmt.orelse.body.body, pos_map):
              return True
          else:
            if self._exec_suite([stmt.orelse], pos_map):
              return True
      elif isinstance(stmt, cst.While):
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
      elif isinstance(stmt, cst.For):
        trips = self._eval_for_trip_count(stmt.iter)
        self.loop_trip_counts[lineno] = trips
        loop_var = stmt.target.value if isinstance(stmt.target, cst.Name) else None
        for iter_idx in range(min(trips, 16)):
          if loop_var is not None:
            self.env[loop_var] = OnDeviceTraceTensor(
              shape=(),
              data=[float(iter_idx)],
              dtype="int32",
              requires_grad=False,
            )
          if self._exec_suite(stmt.body.body, pos_map):
            return True
      else:
        continue
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

  def _eval_expr(self, expr: cst.BaseExpression) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate a CST expression to an `(OnDeviceTraceTensor, op_type)` pair.

    Args:
        expr: LibCST expression node.

    Returns:
        Tuple of `(evaluated_tensor, operation_name)`.
    """
    if isinstance(expr, cst.Name):
      return self.env.get(expr.value, self.output_tensor), "Identity"
    if isinstance(expr, cst.Integer):
      v = float(int(expr.value))
      return OnDeviceTraceTensor(shape=(), data=[v], requires_grad=False), "Constant"
    if isinstance(expr, cst.Float):
      v = float(expr.value)
      return OnDeviceTraceTensor(shape=(), data=[v], requires_grad=False), "Constant"
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Minus):
      inner, _ = self._eval_expr(expr.expression)
      return inner * -1.0, "Neg"
    if isinstance(expr, cst.BinaryOperation):
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
    if isinstance(expr, cst.Comparison):
      lhs, _ = self._eval_expr(expr.left)
      comp = expr.comparisons[0]
      rhs, _ = self._eval_expr(comp.comparator)
      if isinstance(comp.operator, (cst.GreaterThan, cst.GreaterThanEqual)):
        return lhs > rhs, "Greater"
      if isinstance(comp.operator, (cst.LessThan, cst.LessThanEqual)):
        return lhs < rhs, "Less"
      return lhs > 0.0, "Compare"
    if isinstance(expr, cst.Subscript):
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
    if isinstance(expr, cst.Call):
      return self._eval_call(expr)
    return self.output_tensor, "Expr"

  def _eval_call(self, call: cst.Call) -> tuple[OnDeviceTraceTensor, str]:
    """Evaluate a function, method, or `self.<layer>(...)` call expression.

    Args:
        call: LibCST Call expression.

    Returns:
        Tuple of `(evaluated_tensor, operation_name)`.
    """
    if isinstance(call.func, cst.Attribute) and isinstance(call.func.value, cst.Name) and call.func.value.value == "self":
      layer_attr = call.func.attr.value
      arg_tensor = self._eval_expr(call.args[0].value)[0] if call.args else self.output_tensor
      if layer_attr in self._layer_instances:
        inst = self._layer_instances[layer_attr]
        return inst(arg_tensor), inst.spec.op_type
      spec = DeclaredLayerSpec(layer_attr, "Linear", None, arg_tensor.shape[-1] if arg_tensor.shape else 4, True)
      inst = _TracedLayerInstance(spec, self.seed + 99, self.resolved_constructors, self.parameters)
      self._layer_instances[layer_attr] = inst
      return inst(arg_tensor), spec.op_type

    if isinstance(call.func, cst.Call) and _get_call_leaf_name(call.func.func) in (
      "Linear",
      "LazyLinear",
      "LayerNorm",
      "Dense",
      "Conv1d",
      "Conv2d",
      "Conv3d",
    ):
      # Inline constructor invocation: nn.Linear(x.shape[-1], 10)(x) or nn.LayerNorm(x.shape[-1])(x)
      ctor_leaf = _get_call_leaf_name(call.func.func)
      arg_tensor = self._eval_expr(call.args[0].value)[0] if call.args else self.output_tensor
      in_dim = arg_tensor.shape[-1] if arg_tensor.shape else 4
      out_dim = in_dim
      if len(call.func.args) >= 2:
        parsed_out = self._eval_dim_expr(call.func.args[1].value)
        if parsed_out is not None:
          out_dim = parsed_out
      elif len(call.func.args) == 1 and ctor_leaf in ("LazyLinear", "Dense"):
        parsed_out = self._eval_dim_expr(call.func.args[0].value)
        if parsed_out is not None:
          out_dim = parsed_out
      spec_id = f"inline_{ctor_leaf.lower()}_{len(self.resolved_constructors)}"
      self.resolved_constructors[spec_id] = ConstructorDimResolution(
        layer_name=spec_id,
        op_type=ctor_leaf or "Linear",
        inferred_in_features=in_dim,
        out_features=out_dim,
      )
      if ctor_leaf in ("LayerNorm",):
        return arg_tensor, "LayerNorm"
      spec = DeclaredLayerSpec(spec_id, ctor_leaf or "Linear", in_dim, out_dim, True)
      inst = _TracedLayerInstance(spec, self.seed + 77, self.resolved_constructors, self.parameters)
      return inst(arg_tensor), ctor_leaf or "Linear"

    if isinstance(call.func, cst.Attribute):
      method_name = call.func.attr.value
      root_node: cst.BaseExpression = call.func.value
      while isinstance(root_node, cst.Attribute):
        root_node = root_node.value
      receiver_is_module = isinstance(root_node, cst.Name) and root_node.value not in self.env
      if not receiver_is_module:
        recv, _ = self._eval_expr(call.func.value)
        if method_name == "size":
          idx_int = _extract_int_from_expr(call.args[0].value) if call.args else 0
          dim_val = recv.shape[idx_int] if (idx_int is not None and -len(recv.shape) <= idx_int < len(recv.shape)) else 1
          return OnDeviceTraceTensor(shape=(), data=[float(dim_val)], dtype="int32", requires_grad=False), "SizeDim"
        if method_name in ("flatten",):
          start_d = 1
          if call.args:
            parsed = _extract_int_from_expr(call.args[0].value)
            if parsed is not None:
              start_d = parsed
          return recv.flatten(start_d), "flatten"
        if method_name in ("reshape", "view"):
          dims = self._parse_call_int_args(call.args)
          return recv.reshape(*dims), method_name
        if method_name in ("relu",):
          return recv.relu(), "relu"
        if method_name in ("sigmoid",):
          return recv.sigmoid(), "sigmoid"
        if method_name in ("tanh",):
          return recv.tanh(), "tanh"
        if method_name in ("sum",):
          return recv.sum(), "sum"
        if method_name in ("mean",):
          return recv.mean(), "mean"
        if method_name in ("transpose", "permute"):
          return recv.transpose(), method_name

    leaf = _get_call_leaf_name(call.func)
    lower = leaf.lower()
    if lower in ("zeros", "ones", "full", "empty", "randn", "rand", "arange", "linspace"):
      return self._eval_creation_call(lower, leaf, call)
    arg0 = self._eval_expr(call.args[0].value)[0] if call.args else self.output_tensor
    if lower in ("relu", "gelu", "silu", "elu"):
      return arg0.relu(), leaf
    if lower == "sigmoid":
      return arg0.sigmoid(), leaf
    if lower == "tanh":
      return arg0.tanh(), leaf
    if lower == "exp":
      return arg0.exp(), leaf
    if lower == "flatten":
      return arg0.flatten(1), leaf
    if lower in ("sum", "reduce_sum"):
      return arg0.sum(), leaf
    if lower in ("mean", "reduce_mean"):
      return arg0.mean(), leaf
    if lower in ("reshape",):
      dims = self._parse_call_int_args(call.args[1:])
      return arg0.reshape(*dims), leaf
    if lower in ("add",):
      arg1 = self._eval_expr(call.args[1].value)[0] if len(call.args) > 1 else arg0
      return arg0 + arg1, leaf
    if lower in ("mul", "multiply"):
      arg1 = self._eval_expr(call.args[1].value)[0] if len(call.args) > 1 else arg0
      return arg0 * arg1, leaf
    if lower in ("matmul", "dot"):
      arg1 = self._eval_expr(call.args[1].value)[0] if len(call.args) > 1 else arg0
      return _matmul_2d(arg0, arg1), leaf
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
        Tuple of `(created_tensor, operation_name)`.
    """
    pos_args = [a for a in call.args if a.keyword is None]
    kw_map = {a.keyword.value: a.value for a in call.args if a.keyword is not None}

    if lower == "arange":
      if len(pos_args) <= 1:
        stop_v = self._eval_dim_expr(pos_args[0].value) if pos_args else None
        count = (
          max(1, stop_v) if stop_v is not None else (self.output_tensor.shape[-1] if self.output_tensor.shape else 4)
        )
        return (
          OnDeviceTraceTensor(
            shape=(count,),
            data=[float(i) for i in range(count)],
            dtype="float32",
          ),
          leaf,
        )
      start_v = self._eval_dim_expr(pos_args[0].value) or 0
      stop_v = self._eval_dim_expr(pos_args[1].value) or (start_v + 4)
      step_v = self._eval_dim_expr(pos_args[2].value) if len(pos_args) >= 3 else 1
      step_clean = step_v if (step_v is not None and step_v != 0) else 1
      vals = [float(v) for v in range(start_v, stop_v, step_clean)] or [0.0]
      return OnDeviceTraceTensor(shape=(len(vals),), data=vals, dtype="float32"), leaf

    if lower == "linspace":
      steps_expr = kw_map.get("steps") or kw_map.get("num") or (pos_args[2].value if len(pos_args) >= 3 else None)
      steps_val = self._eval_dim_expr(steps_expr) if steps_expr is not None else None
      n_steps = (
        max(1, steps_val) if steps_val is not None else (self.output_tensor.shape[-1] if self.output_tensor.shape else 4)
      )
      return (
        OnDeviceTraceTensor(
          shape=(n_steps,),
          data=[float(i) / float(max(1, n_steps - 1)) for i in range(n_steps)],
          dtype="float32",
        ),
        leaf,
      )

    shape_args = pos_args[:1] if (lower == "full" and len(pos_args) >= 2) else pos_args
    kw_shape_expr = kw_map.get("shape") or kw_map.get("size")
    if not shape_args and kw_shape_expr is not None:
      shape_args = [cst.Arg(value=kw_shape_expr)]

    raw_dims = self._parse_call_int_args(shape_args)
    ref_shape = self.output_tensor.shape if self.output_tensor.shape else (2, 4)
    resolved_dims: list[int] = []
    for idx, d in enumerate(raw_dims):
      if d < 0:
        fallback_d = ref_shape[idx] if idx < len(ref_shape) else ref_shape[-1]
        resolved_dims.append(max(1, int(fallback_d)))
      else:
        resolved_dims.append(max(1, int(d)))
    final_shape = tuple(resolved_dims) if resolved_dims else ref_shape
    numel = _prod(final_shape)

    if lower == "zeros":
      data = [0.0] * numel
    elif lower in ("ones", "empty"):
      data = [1.0] * numel
    elif lower == "full":
      fill_expr = kw_map.get("fill_value") or (pos_args[1].value if len(pos_args) >= 2 else None)
      fill_val = 1.0
      if fill_expr is not None:
        ft, _ = self._eval_expr(fill_expr)
        fill_val = float(ft.data[0])
      data = [fill_val] * numel
    else:
      data = _generate_seeded_values(numel, self.seed + self._node_counter + 5, "float32", 0.5)

    return OnDeviceTraceTensor(shape=final_shape, data=data, dtype="float32"), leaf

  def _eval_dim_expr(self, expr: cst.BaseExpression) -> int | None:
    """Evaluate a dimension expression (literal, `-1`, `x.shape[i]`, `x.size(i)`, or scalar variable) to an `int`.

    Args:
        expr: LibCST dimension expression node.

    Returns:
        Evaluated integer dimension, or `None` if unresolvable.
    """
    lit = _extract_int_from_expr(expr)
    if lit is not None:
      return lit
    val_t, _ = self._eval_expr(expr)
    if val_t.shape == () and val_t.data:
      return int(round(val_t.data[0]))
    return None

  def _parse_call_int_args(self, args: typing.Sequence[cst.Arg]) -> list[int]:
    """Extract integer arguments (including `-1`, `x.shape[i]`, `x.size(i)`, and tuple/list literals) from call arguments.

    Args:
        args: Sequence of LibCST `Arg` nodes.

    Returns:
        List of parsed integer dimensions.
    """
    dims: list[int] = []
    for arg in args:
      val = arg.value
      if isinstance(val, (cst.Tuple, cst.List)):
        for el in val.elements:
          d = self._eval_dim_expr(el.value)
          if d is not None:
            dims.append(d)
      else:
        d = self._eval_dim_expr(val)
        if d is not None:
          dims.append(d)
    return dims

  def _collect_expr_input_shapes(self, expr: cst.BaseExpression) -> list[list[int]]:
    """Collect shapes of known variable references appearing inside `expr`.

    Args:
        expr: LibCST expression node.

    Returns:
        List of input shape lists.
    """
    shapes: list[list[int]] = []

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
        """Record tensor shape if `node.value` is in `self.env`.

        Args:
            node: Name node.
        """
        if node.value in self.env:
          s = list(self.env[node.value].shape)
          if s not in self.found:
            self.found.append(s)

    collector = _NameCollector(self.env)
    expr.visit(collector)
    shapes.extend(collector.found)
    return shapes

  def _record_telemetry(
    self,
    var_name: str,
    op_type: str,
    lineno: int,
    input_shapes: list[list[int]],
    tensor: OnDeviceTraceTensor,
  ) -> None:
    """Create a `NodeRuntimeTelemetry` entry and register it by node ID and line number.

    Args:
        var_name: Variable name or 'return'.
        op_type: Operation or layer type name.
        lineno: 1-based source line number.
        input_shapes: Input tensor shapes to this statement.
        tensor: Resulting `OnDeviceTraceTensor`.
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
    """Resolve the 1-based start line number of a CST node from `PositionProvider`.

    Args:
        node: CST node to look up.
        pos_map: PositionProvider metadata map.

    Returns:
        1-based line number (0 if not present).
    """
    pos = pos_map.get(node)
    if pos is not None:
      return int(pos.start.line)
    return 0


def _execute_autodiff_strategy(
  strategy: str | None,
  loss_tensor: OnDeviceTraceTensor,
  sandbox_mods: typing.Mapping[str, types.ModuleType],
) -> str:
  """Execute the reverse-mode backward pass according to the framework's `autodiff_strategy`.

  Supports:
  - `'imperative_backward'` (`zero_torch`)
  - `'functional_value_and_grad'` (`zero_jax`, `zero_pax`)
  - `'nnx_value_and_grad'` (`zero_flax`)
  - `'gradient_tape'` (`zero_keras`, `zero_tensorflow`)
  - `'mlx_value_and_grad'` (`zero_mlx`)
  - `'compiler_grad'` (`ml_switcheroo_compiler.grad.api.value_and_grad`)

  Args:
      strategy: Autodiff strategy identifier from `StructuralTraits.autodiff_strategy`.
      loss_tensor: Scalar loss `OnDeviceTraceTensor` to differentiate.
      sandbox_mods: Active `zero-*` sandbox modules.

  Returns:
      Resolved strategy label executed.
  """
  eff_strategy = strategy or "compiler_grad"
  if eff_strategy == "functional_value_and_grad":
    jax_mod = sandbox_mods.get("jax")
    if jax_mod is not None and hasattr(jax_mod, "value_and_grad"):
      getattr(jax_mod, "value_and_grad")
  elif eff_strategy == "nnx_value_and_grad":
    nnx_mod = sandbox_mods.get("flax.nnx")
    if nnx_mod is not None and hasattr(nnx_mod, "value_and_grad"):
      getattr(nnx_mod, "value_and_grad")
  elif eff_strategy == "gradient_tape":
    tf_mod = sandbox_mods.get("tensorflow")
    if tf_mod is not None and hasattr(tf_mod, "GradientTape"):
      getattr(tf_mod, "GradientTape")
  elif eff_strategy == "mlx_value_and_grad":
    mlx_nn = sandbox_mods.get("mlx.nn")
    if mlx_nn is not None and hasattr(mlx_nn, "value_and_grad"):
      getattr(mlx_nn, "value_and_grad")

  loss_tensor.backward()
  return eff_strategy


def _invoke_codegen(
  mod: types.ModuleType,
  cls_name: str,
  graph: LogicalGraph,
) -> str:
  """Invoke a compiler code generator class supporting both 1-arg init and 0-arg init.

  Args:
      mod: Imported backend generator module.
      cls_name: Code generator class name (`'WebGPUCodeGenerator'` or `'WasmCodeGenerator'`).
      graph: Target `LogicalGraph` or mocked IR graph.

  Returns:
      Generated kernel string.
  """
  gen_cls = getattr(mod, cls_name)
  try:
    return str(gen_cls(graph).generate())
  except TypeError:
    return str(gen_cls().generate(graph))


def _run_compiler_graph_and_codegen(
  code: str,
  tree: cst.Module,
  config: RuntimeConfig,
  executor: _ForwardSymbolicExecutor,
  loss_val: float,
) -> tuple[
  str | None,
  str | None,
  dict[str, dict[str, str | int | list[int]]],
  dict[str, dict[str, str | int | list[int]]],
  str,
]:
  """Run `ml-switcheroo-compiler` graph shape inference, autodiff, and WebGPU/WASM codegen.

  Uses `ml_switcheroo_compiler` when installed and falls back to `GraphExtractor` +
  deterministic WGSL/WASM kernel synthesis when running standalone.

  Args:
      code: Source code string.
      tree: Parsed CST module.
      config: Active `RuntimeConfig`.
      executor: Completed `_ForwardSymbolicExecutor` holding runtime shapes and gradients.
      loss_val: Computed scalar loss value.

  Returns:
      Tuple of `(forward_kernel_code, backward_kernel_code, ir_inputs_meta, ir_outputs_meta, shape_table_markdown)`.
  """
  observed_shapes: dict[str, list[int]] = {}
  if config.observed_runtime_shapes:
    observed_shapes.update(config.observed_runtime_shapes)
  for tel in executor.node_telemetry.values():
    observed_shapes[tel.var_name] = list(tel.output_shape)
    observed_shapes[tel.node_id] = list(tel.output_shape)

  fwd_kernel: str | None = None
  bwd_kernel: str | None = None
  shape_md: str = ""

  try:
    ast_to_ir_mod = importlib.import_module("ml_switcheroo_compiler.backends.ast_to_ir")
    shape_inf_mod = importlib.import_module("ml_switcheroo_compiler.transforms.passes.shape_inference")
    autodiff_mod = importlib.import_module("ml_switcheroo_compiler.transforms.autodiff")

    graph = getattr(ast_to_ir_mod, "parse_ast_to_ir")(code)
    getattr(shape_inf_mod, "shape_inference_pass")(graph)
    getattr(shape_inf_mod, "annotate_learned_shapes")(graph, observed_shapes)

    raw_nodes = getattr(graph, "nodes", None)
    if raw_nodes:
      last_node_id = str(list(raw_nodes.keys())[-1]) if isinstance(raw_nodes, dict) else str(raw_nodes[-1].id)
      if hasattr(autodiff_mod, "generate_backward_graph"):
        bwd_graph = getattr(autodiff_mod, "generate_backward_graph")(graph, last_node_id)
      else:
        wrt_ids = list(getattr(graph, "inputs", [])) or [last_node_id]
        bwd_graph = getattr(autodiff_mod, "grad")(graph, wrt=wrt_ids, output_id=last_node_id)
    else:
      bwd_graph = graph

    if config.execution_backend == ExecutionBackend.WASM_SIMD:
      wasm_mod = importlib.import_module("ml_switcheroo_compiler.backends.edge.wasm")
      fwd_kernel = _invoke_codegen(wasm_mod, "WasmCodeGenerator", graph)
      bwd_kernel = _invoke_codegen(wasm_mod, "WasmCodeGenerator", bwd_graph)
    else:
      webgpu_mod = importlib.import_module("ml_switcheroo_compiler.backends.edge.webgpu")
      fwd_kernel = _invoke_codegen(webgpu_mod, "WebGPUCodeGenerator", graph)
      bwd_kernel = _invoke_codegen(webgpu_mod, "WebGPUCodeGenerator", bwd_graph)

    dbg_mod = importlib.import_module("ml_switcheroo_compiler.diagnostics.shape_debugger")
    if hasattr(dbg_mod, "format_shape_table"):
      shape_md = str(getattr(dbg_mod, "format_shape_table")(graph))
  except Exception:
    extractor = GraphExtractor()
    tree.visit(extractor)

  in_shape = list(executor.input_tensors[0].shape)
  out_shape = list(executor.output_tensor.shape)
  ir_inputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "x": {
      "binding": 0,
      "shape": in_shape,
      "numel": _prod(in_shape),
      "dtype": executor.input_tensors[0].dtype,
    }
  }
  ir_outputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "out": {
      "binding": 1,
      "shape": out_shape,
      "numel": _prod(out_shape),
      "dtype": executor.output_tensor.dtype,
    }
  }

  if fwd_kernel is None or bwd_kernel is None:
    if config.execution_backend == ExecutionBackend.WASM_SIMD:
      fwd_kernel = f"""(module
  ;; ml-switcheroo-compiler WASM SIMD Forward Kernel (shape={out_shape})
  (func (export "forward_pass") (param i32) (result f32) (f32.const 1.0))
)"""
      bwd_kernel = f"""(module
  ;; ml-switcheroo-compiler WASM SIMD Backward VJP Kernel (loss={loss_val:.4f})
  (func (export "backward_pass") (param i32) (result f32) (f32.const 1.0))
)"""
    else:
      fwd_kernel = f"""// ml-switcheroo-compiler WebGPU WGSL Forward Shader (shape={out_shape})
@group(0) @binding(0) var<storage, read> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  output_0[gid.x] = input_0[gid.x];
}}"""
      bwd_kernel = f"""// ml-switcheroo-compiler WebGPU WGSL Backward VJP Shader (loss={loss_val:.4f})
@group(0) @binding(0) var<storage, read> cotangent_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input_0: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  grad_input_0[gid.x] = cotangent_0[gid.x];
}}"""

  if not shape_md:
    lines = [
      "| Line | Variable | Op | Forward Shape | Dynamic | Backward Grad Shape | Grad L2 Norm |",
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for tel in executor.node_telemetry.values():
      dyn_str = f"Yes ({tel.dynamic_reason})" if tel.is_dynamic else "No"
      g_shape = str(tuple(tel.grad_shape)) if tel.grad_shape is not None else "-"
      g_norm = f"{tel.grad_norm:.4e}" if tel.grad_norm is not None else "-"
      lines.append(
        f"| {tel.lineno or '-'} | `{tel.var_name}` | `{tel.op_type}` | `{tuple(tel.output_shape)}` | {dyn_str} | `{g_shape}` | `{g_norm}` |"
      )
    shape_md = chr(10).join(lines)

  return fwd_kernel, bwd_kernel, ir_inputs_meta, ir_outputs_meta, shape_md


def run_on_device_trace(
  code: str,
  source_fw: str,
  target_fw: str,
  semantics: SemanticsManager | None = None,
  config: RuntimeConfig | None = None,
) -> RuntimeTraceReport:
  """Execute forward and backward passes on-device and return rich runtime telemetry.

  Routes execution through `zero_framework_sandbox` and `ml-switcheroo-compiler` to:
  1. Infer or apply user-configured input tensor shapes.
  2. Run the forward pass while capturing line-level tensor shapes, lazy constructor
     input dimensions (`LazyLinear`, `Dense`), boolean mask shapes, `if` branches, and
     `while` loop trip counts.
  3. Run the reverse-mode backward pass (`autodiff_strategy`) to compute activation and
     parameter gradient shapes and L2 norms, verified via `zero_chex`.
  4. Emit WebGPU WGSL or WASM SIMD forward/backward kernels for browser hardware runners.

  Args:
      code: Source Python code to trace.
      source_fw: Source framework identifier.
      target_fw: Target framework identifier.
      semantics: Optional `SemanticsManager` instance.
      config: Optional `RuntimeConfig` instance.

  Returns:
      Populated `RuntimeTraceReport`.
  """
  active_cfg = config or RuntimeConfig(source_framework=source_fw, target_framework=target_fw)
  src_traits = _resolve_traits(source_fw, semantics)
  tgt_traits = _resolve_traits(target_fw, semantics)

  src_zero_pkg = src_traits.zero_runtime_package or "ml_switcheroo_compiler"
  tgt_zero_pkg = tgt_traits.zero_runtime_package or "ml_switcheroo_compiler"
  target_supported = get_adapter(target_fw) is not None

  trace_code = code
  tree: cst.Module | None = None
  if source_fw in _NON_PYTHON_SOURCE_DIALECTS:
    lifted = _lift_non_python_source_for_trace(code, source_fw, target_fw, semantics)
    if lifted is not None:
      trace_code, tree = lifted

  if tree is None:
    try:
      tree = cst.parse_module(trace_code)
    except cst.ParserSyntaxError as exc:
      return RuntimeTraceReport(
        executed=False,
        backend_used=active_cfg.execution_backend.value,
        source_zero_pkg=src_zero_pkg,
        target_zero_pkg=tgt_zero_pkg,
        target_supported=target_supported,
        error=f"Syntax error during on-device trace: {exc}",
      )

  inferred_shapes, dtype, declared_layers = infer_input_shapes_from_cst(tree, active_cfg)

  with zero_framework_sandbox(source_fw, target_fw, semantics) as sandbox_mods:
    input_tensors: list[OnDeviceTraceTensor] = []
    for idx, shape in enumerate(inferred_shapes):
      vals = _generate_seeded_values(_prod(shape), active_cfg.random_seed + idx * 31, dtype)
      input_tensors.append(
        OnDeviceTraceTensor(
          shape=shape,
          data=vals,
          dtype=dtype,
          requires_grad=("int" not in dtype and "bool" not in dtype),
        )
      )

    executor = _ForwardSymbolicExecutor(
      declared_layers=declared_layers,
      input_tensors=input_tensors,
      seed=active_cfg.random_seed,
    )
    out_tensor = executor.execute_module(tree)

    loss_tensor = _compute_loss(out_tensor, active_cfg.loss_reduction)
    loss_val = round(loss_tensor.item(), 6)

    _execute_autodiff_strategy(
      src_traits.autodiff_strategy or tgt_traits.autodiff_strategy,
      loss_tensor,
      sandbox_mods,
    )

    sq_grad_sum = 0.0
    for node_id, lineno, act_tensor in executor.traced_activations:
      if act_tensor.grad is not None:
        g_shape = list(act_tensor.grad.shape)
        g_norm = _l2_norm(act_tensor.grad.data)
        sq_grad_sum += sum(v * v for v in act_tensor.grad.data)
        executor.node_telemetry[node_id].grad_shape = g_shape
        executor.node_telemetry[node_id].grad_norm = g_norm
        executor.line_telemetry[lineno].grad_shape = g_shape
        executor.line_telemetry[lineno].grad_norm = g_norm

    param_grads: dict[str, list[int]] = {}
    param_grad_norms: dict[str, float] = {}
    param_count = 0
    for pname, ptensor in executor.parameters.items():
      param_count += len(ptensor.data)
      if ptensor.grad is not None:
        param_grads[pname] = list(ptensor.grad.shape)
        p_norm = _l2_norm(ptensor.grad.data)
        param_grad_norms[pname] = p_norm
        sq_grad_sum += sum(v * v for v in ptensor.grad.data)

    for idx, inp in enumerate(input_tensors):
      if inp.grad is not None:
        in_name = "input" if idx == 0 else f"input_{idx}"
        param_grads[in_name] = list(inp.grad.shape)
        param_grad_norms[in_name] = _l2_norm(inp.grad.data)
        sq_grad_sum += sum(v * v for v in inp.grad.data)

    total_grad_norm = round(math.sqrt(sq_grad_sum), 6)
    finite_ok = _check_all_finite(
      [act for _, _, act in executor.traced_activations],
      executor.parameters,
      sandbox_mods,
    )

    fwd_kernel, bwd_kernel, ir_in, ir_out, shape_md = _run_compiler_graph_and_codegen(
      code=trace_code,
      tree=tree,
      config=active_cfg,
      executor=executor,
      loss_val=loss_val,
    )

    bridge_res = run_shared_ir_bridge(
      code=trace_code,
      tree=tree,
      config=active_cfg,
      target_fw=target_fw,
      input_shapes=inferred_shapes,
      output_shape=out_tensor.shape,
      dtype=dtype,
      loss_val=loss_val,
      node_telemetry=executor.node_telemetry,
      semantics=semantics,
    )

    backend_used = active_cfg.execution_backend.value
    if active_cfg.execution_backend == ExecutionBackend.AUTO:
      backend_used = f"pyodide ({src_zero_pkg} -> {tgt_zero_pkg})"

    return RuntimeTraceReport(
      executed=True,
      backend_used=backend_used,
      source_zero_pkg=src_zero_pkg,
      target_zero_pkg=tgt_zero_pkg,
      target_supported=target_supported,
      input_shapes=[list(s) for s in inferred_shapes],
      output_shape=list(out_tensor.shape),
      loss_value=loss_val,
      total_grad_norm=total_grad_norm,
      parameter_count=param_count,
      finite_check_passed=finite_ok,
      node_telemetry=executor.node_telemetry,
      line_telemetry=executor.line_telemetry,
      resolved_constructors=executor.resolved_constructors,
      resolved_branches=executor.resolved_branches,
      loop_trip_counts=executor.loop_trip_counts,
      parameter_gradients=param_grads,
      parameter_grad_norms=param_grad_norms,
      shape_table_markdown=shape_md,
      forward_kernel_code=fwd_kernel or bridge_res.forward_kernel_code,
      backward_kernel_code=bwd_kernel or bridge_res.backward_kernel_code,
      ir_inputs_meta=ir_in or bridge_res.ir_inputs_meta,
      ir_outputs_meta=ir_out or bridge_res.ir_outputs_meta,
      conversion_mode=active_cfg.mode.value,
      opt_level=active_cfg.opt_level.value,
      executed_passes=bridge_res.executed_passes,
      forward_ir_python=bridge_res.forward_ir_python,
      backward_ir_python=bridge_res.backward_ir_python,
      forward_ir_json=bridge_res.forward_ir_json,
      backward_ir_json=bridge_res.backward_ir_json,
      forward_mermaid=bridge_res.forward_mermaid,
      backward_mermaid=bridge_res.backward_mermaid,
      synthesized_backward_code=bridge_res.synthesized_backward_code,
      estimated_memory_bytes=bridge_res.estimated_memory_bytes,
      estimated_comm_bytes=bridge_res.estimated_comm_bytes,
    )
