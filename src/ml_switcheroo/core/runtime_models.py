"""Strongly-typed Pydantic models for on-device forward and backward pass telemetry.

This module defines the structured trace report and node/line telemetry models
produced by `runtime_tracer` when executing models on-device via `zero-*` and
`ml-switcheroo-compiler`.
"""

import libcst as cst
from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo.config import RuntimeConfig


class NodeRuntimeTelemetry(BaseModel):
  """Runtime shape, dtype, and backward gradient telemetry for a single operation or line."""

  model_config = ConfigDict(extra="forbid")

  node_id: str = Field(description="Unique identifier of the traced node or statement.")
  var_name: str = Field(description="Assigned variable name or 'return' for return statements.")
  op_type: str = Field(description="Operation or layer type name.")
  lineno: int | None = Field(default=None, description="1-based source line number if available.")
  input_shapes: list[list[int]] = Field(
    default_factory=list,
    description="Concrete input tensor shapes observed during the forward pass.",
  )
  output_shape: list[int] = Field(
    default_factory=list,
    description="Concrete output tensor shape observed during the forward pass.",
  )
  dtype: str = Field(default="float32", description="Data type of the output tensor.")
  grad_shape: list[int] | None = Field(
    default=None,
    description="Concrete gradient shape with respect to this tensor during the backward pass.",
  )
  grad_norm: float | None = Field(
    default=None,
    description="L2 norm of the gradient tensor during the backward pass.",
  )
  is_dynamic: bool = Field(
    default=False,
    description="True if the output shape is data-dependent or unknowable statically.",
  )
  dynamic_reason: str | None = Field(
    default=None,
    description="Explanation of why static analysis could not determine this shape.",
  )


class BranchResolutionRecord(BaseModel):
  """Record of a data-dependent conditional branch resolved during on-device execution."""

  model_config = ConfigDict(extra="forbid")

  lineno: int = Field(description="1-based source line number of the conditional statement.")
  condition_src: str = Field(description="Source expression of the branch condition.")
  taken_branch: str = Field(description="Branch taken at runtime ('true' or 'false').")


class ConstructorDimResolution(BaseModel):
  """Runtime-resolved input dimension for a lazy or unbuilt neural layer constructor."""

  model_config = ConfigDict(extra="forbid")

  layer_name: str = Field(description="Attribute name of the layer on self (e.g. 'fc1').")
  op_type: str = Field(description="Layer class name (e.g. 'LazyLinear', 'Linear', 'Dense', 'Conv2d').")
  inferred_in_features: int = Field(description="Input feature/channel dimension observed on first forward call.")
  out_features: int | None = Field(default=None, description="Output feature/channel dimension if known.")


class DeclaredLayerSpec:
  """Metadata for a layer attribute declared in `__init__` or `setup`."""

  def __init__(
    self,
    layer_name: str,
    op_type: str,
    in_features: int | None,
    out_features: int | None,
    is_lazy: bool,
  ) -> None:
    """Initialize declared layer specification.

    Args:
        layer_name: Attribute name on `self`.
        op_type: Layer class name.
        in_features: Static input feature count if known.
        out_features: Static output feature count if known.
        is_lazy: True if input dimension is deferred to runtime.
    """
    self.layer_name = layer_name
    self.op_type = op_type
    self.in_features = in_features
    self.out_features = out_features
    self.is_lazy = is_lazy


class RuntimeTraceReport(BaseModel):
  """Complete report of an on-device forward and backward execution pass."""

  model_config = ConfigDict(extra="forbid")

  executed: bool = Field(default=False, description="True if on-device execution completed.")
  backend_used: str = Field(default="pyodide", description="Execution backend used ('pyodide', 'webgpu', 'wasm_simd').")
  source_zero_pkg: str = Field(default="", description="Source zero-* runtime package name.")
  target_zero_pkg: str = Field(default="", description="Target zero-* runtime package name.")
  target_supported: bool = Field(default=True, description="True if target framework is supported for emission.")
  input_shapes: list[list[int]] = Field(
    default_factory=list,
    description="Input tensor shapes fed into the forward pass.",
  )
  output_shape: list[int] = Field(
    default_factory=list,
    description="Final output tensor shape produced by the forward pass.",
  )
  loss_value: float | None = Field(
    default=None,
    description="Scalar loss value computed from the forward output.",
  )
  total_grad_norm: float | None = Field(
    default=None,
    description="Global L2 norm across all computed backward gradients.",
  )
  parameter_count: int = Field(
    default=0,
    description="Total number of scalar parameters discovered in the model.",
  )
  finite_check_passed: bool = Field(
    default=True,
    description="True if all forward activations and backward gradients are finite (no NaN/Inf).",
  )
  node_telemetry: dict[str, NodeRuntimeTelemetry] = Field(
    default_factory=dict,
    description="Mapping from node ID to runtime telemetry.",
  )
  line_telemetry: dict[int, NodeRuntimeTelemetry] = Field(
    default_factory=dict,
    description="Mapping from 1-based source line number to runtime telemetry.",
  )
  resolved_constructors: dict[str, ConstructorDimResolution] = Field(
    default_factory=dict,
    description="Mapping from layer attribute name to runtime-resolved input dimensions.",
  )
  resolved_branches: list[BranchResolutionRecord] = Field(
    default_factory=list,
    description="List of data-dependent conditional branches resolved during execution.",
  )
  loop_trip_counts: dict[int, int] = Field(
    default_factory=dict,
    description="Mapping from while-loop line number to executed iteration count.",
  )
  parameter_gradients: dict[str, list[int]] = Field(
    default_factory=dict,
    description="Mapping from parameter name to backward gradient shape.",
  )
  parameter_grad_norms: dict[str, float] = Field(
    default_factory=dict,
    description="Mapping from parameter name to backward gradient L2 norm.",
  )
  shape_table_markdown: str = Field(
    default="",
    description="Formatted Markdown table summarizing tensor shapes and gradients.",
  )
  forward_kernel_code: str | None = Field(
    default=None,
    description="Compiled WebGPU WGSL or WASM SIMD kernel source for the forward pass.",
  )
  backward_kernel_code: str | None = Field(
    default=None,
    description="Compiled WebGPU WGSL or WASM SIMD kernel source for the backward pass.",
  )
  ir_inputs_meta: dict[str, dict[str, str | int | list[int]]] = Field(
    default_factory=dict,
    description="Input tensor buffer metadata for browser WebGPU/WASM runners.",
  )
  ir_outputs_meta: dict[str, dict[str, str | int | list[int]]] = Field(
    default_factory=dict,
    description="Output tensor buffer metadata for browser WebGPU/WASM runners.",
  )
  conversion_mode: str = Field(
    default="preserve_trivia",
    description="Active conversion mode ('preserve_trivia' or 'optimized').",
  )
  opt_level: str = Field(
    default="O2",
    description="Active optimization level ('O0', 'O1', 'O2', 'O3').",
  )
  executed_passes: list[str] = Field(
    default_factory=list,
    description="Compiler transformation passes executed on the shared LogicalGraph.",
  )
  forward_ir_python: str = Field(
    default="",
    description="Serialized ml-switcheroo-ir Python representation of forward_graph.",
  )
  backward_ir_python: str = Field(
    default="",
    description="Serialized ml-switcheroo-ir Python representation of backward_graph.",
  )
  forward_ir_json: str = Field(
    default="",
    description="Serialized JSON representation of forward_graph.",
  )
  backward_ir_json: str = Field(
    default="",
    description="Serialized JSON representation of backward_graph.",
  )
  forward_mermaid: str = Field(
    default="",
    description="Mermaid DAG representation of forward_graph with concrete TensorSpec edge labels.",
  )
  backward_mermaid: str = Field(
    default="",
    description="Mermaid DAG representation of backward_graph with concrete TensorSpec edge labels.",
  )
  synthesized_backward_code: str = Field(
    default="",
    description="Target-framework Python backward/VJP code synthesized from backward_graph.",
  )
  estimated_memory_bytes: float = Field(
    default=0.0,
    description="Estimated total memory footprint in bytes (parameters + activations + gradients).",
  )
  estimated_comm_bytes: float = Field(
    default=0.0,
    description="Estimated collective communication volume in bytes.",
  )
  error: str | None = Field(
    default=None,
    description="Error message if on-device execution encountered a failure.",
  )


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
