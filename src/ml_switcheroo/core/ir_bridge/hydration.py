"""In-Place Runtime Trace Hydration for ``LogicalGraph`` Nodes in ``ir_bridge``.

Provides ``_format_node_runtime_comment`` and ``hydrate_graph_with_runtime_trace``.
"""

from __future__ import annotations

import contextlib

from ml_switcheroo_ir import LogicalGraph, LogicalNode, TensorSpec

from ml_switcheroo.core.ir_bridge.proxy_tracing import _resolve_ir_dtype
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry, RuntimeTraceReport
from ml_switcheroo.core.runtime_tensor import _prod


def _format_node_runtime_comment(
  output_shape: list[int] | tuple[int, ...],
  dtype: str,
  grad_shape: list[int] | None = None,
  grad_norm: float | None = None,
  dynamic_reason: str | None = None,
) -> str:
  """Format a concise runtime telemetry comment for a ``LogicalNode``.

  Args:
      output_shape: Forward pass output tensor shape.
      dtype: Tensor data type string.
      grad_shape: Optional backward gradient tensor shape.
      grad_norm: Optional backward gradient L2 norm.
      dynamic_reason: Optional explanation for dynamic shape resolution.

  Returns:
      Formatted comment string.
  """
  parts = [f"[on-device fwd] shape={tuple(output_shape)} dtype={dtype}"]
  if dynamic_reason:
    parts.append(f"dynamic: {dynamic_reason}")
  if grad_shape is not None and grad_norm is not None:
    parts.append(f"[bwd] grad={tuple(grad_shape)} norm={grad_norm:.4e}")
  return " | ".join(parts)


def _resolve_node_shapes(
  nid: str,
  graph: LogicalGraph,
  incoming_map: dict[str, list[str]],
  matched_tel: NodeRuntimeTelemetry | None,
  trace: RuntimeTraceReport,
  default_in_shape: list[int],
) -> tuple[list[int], list[int]]:
  """Resolve ``(pred_shape, node_out_shape)`` for ``nid`` from incoming edges and telemetry.

  Args:
      nid: Node identifier.
      graph: Target ``LogicalGraph``.
      incoming_map: Precomputed target-to-sources adjacency list.
      matched_tel: Matched ``NodeRuntimeTelemetry`` record, if any.
      trace: Active ``RuntimeTraceReport``.
      default_in_shape: Fallback input shape list.

  Returns:
      Tuple of ``(predecessor_shape, node_output_shape)``.
  """
  pred_ids = incoming_map.get(nid, [])
  pred_meta = graph.nodes[pred_ids[0]].shape_metadata if (pred_ids and pred_ids[0] in graph.nodes) else None
  if isinstance(pred_meta, (list, tuple)) and pred_meta:
    pred_shape = [int(d) for d in pred_meta if isinstance(d, int)]
  elif matched_tel is not None and matched_tel.input_shapes:
    pred_shape = list(matched_tel.input_shapes[0])
  else:
    pred_shape = list(default_in_shape)

  node_out_shape = (
    list(matched_tel.output_shape)
    if matched_tel is not None and matched_tel.output_shape
    else (list(trace.output_shape) if trace.output_shape else list(pred_shape))
  )
  return pred_shape, node_out_shape


def _hydrate_ctor_and_linear_attrs(
  nid: str,
  node: LogicalNode,
  op_lower: str,
  pred_shape: list[int],
  node_out_shape: list[int],
  trace: RuntimeTraceReport,
) -> None:
  """Populate ``in_features``/``out_features`` or ``in_channels``/``out_channels`` on ``node``.

  Args:
      nid: Node identifier.
      node: ``LogicalNode`` mutated in place.
      op_lower: Lowercased ``node.op_type``.
      pred_shape: Predecessor tensor shape.
      node_out_shape: Mutable output shape list for ``node``.
      trace: Active ``RuntimeTraceReport``.
  """
  ctor = trace.resolved_constructors.get(nid)
  if ctor is not None:
    if "conv" in op_lower:
      node.attributes["in_channels"] = ctor.inferred_in_features
      if ctor.out_features is not None:
        node.attributes["out_channels"] = ctor.out_features
    else:
      node.attributes["in_features"] = ctor.inferred_in_features
      out_val = node.attributes.get("out_features", ctor.out_features)
      if out_val is not None:
        node.attributes["out_features"] = int(str(out_val))
        node_out_shape[-1] = int(str(out_val))
      node.attributes.pop("arg_0", None)
      node.attributes.pop("arg_1", None)
    return

  if not any(k in op_lower for k in ("linear", "dense", "gemm", "matmul")):
    return

  in_feat = pred_shape[-1] if pred_shape else 128
  if "in_features" not in node.attributes:
    if "arg_0" in node.attributes and "arg_1" in node.attributes:
      try:
        node.attributes["in_features"] = int(str(node.attributes.pop("arg_0")))
        node.attributes["out_features"] = int(str(node.attributes.pop("arg_1")))
      except ValueError:
        node.attributes["in_features"] = in_feat
    else:
      raw_arg0 = node.attributes.pop("arg_0", None)
      node.attributes["in_features"] = in_feat
      with contextlib.suppress(ValueError, TypeError):
        node.attributes["out_features"] = int(str(raw_arg0))
  node.attributes.setdefault("out_features", node_out_shape[-1])


def _hydrate_spatial_conv_attrs(
  node: LogicalNode,
  op_lower: str,
  pred_shape: list[int],
  node_out_shape: list[int],
) -> None:
  """Populate channel and spatial byte-stride attributes for convolution/pooling nodes.

  Args:
      node: ``LogicalNode`` mutated in place.
      op_lower: Lowercased ``node.op_type``.
      pred_shape: Predecessor tensor shape.
      node_out_shape: Resolved output shape list.
  """
  if not (any(k in op_lower for k in ("conv", "pool")) and len(pred_shape) >= 3):
    return
  in_c = pred_shape[1]
  in_h = pred_shape[-2] if len(pred_shape) >= 4 else pred_shape[-1]
  in_w = pred_shape[-1]
  node.attributes.setdefault("in_channels", in_c)
  node.attributes.setdefault("out_channels", node_out_shape[1] if len(node_out_shape) >= 2 else in_c)
  node.attributes["input_height"] = in_h
  node.attributes["input_width"] = in_w
  node.attributes["row_stride_bytes"] = in_w * 4
  node.attributes["slice_stride_bytes"] = in_h * in_w * 4


def hydrate_graph_with_runtime_trace(
  graph: LogicalGraph,
  trace: RuntimeTraceReport,
) -> None:
  """Hydrate a ``LogicalGraph`` in-place with runtime-discovered shapes, dimensions, and telemetry.

  Args:
      graph: The ``LogicalGraph`` to hydrate in-place.
      trace: The ``RuntimeTraceReport`` produced by ``run_on_device_trace``.
  """
  if not trace.executed:
    return

  tel_list = list(trace.node_telemetry.values())
  default_in_shape = trace.input_shapes[0] if trace.input_shapes else [2, 4]
  max_loop_trips = max(trace.loop_trip_counts.values()) if trace.loop_trip_counts else None

  incoming_map: dict[str, list[str]] = {}
  for edge in graph.edges:
    incoming_map.setdefault(edge.target, []).append(edge.source)

  non_io_idx = 0
  for nid, node in graph.nodes.items():
    op_lower = str(node.op_type).lower()
    ir_dt = _resolve_ir_dtype("float32")

    if node.op_type in ("Input", "Output"):
      io_shape = (
        tuple(default_in_shape)
        if node.op_type == "Input"
        else (tuple(trace.output_shape) if trace.output_shape else tuple(default_in_shape))
      )
      node.shape_metadata = io_shape
      node.dtype = ir_dt
      node.output_specs = [TensorSpec(shape=io_shape, dtype=ir_dt)]
      node.attributes["num_elements"] = _prod(io_shape)
      continue

    matched_tel: NodeRuntimeTelemetry | None = trace.node_telemetry.get(nid)
    if matched_tel is None:
      matched_tel = next(
        (t for t in tel_list if t.var_name == nid or t.op_type.lower() == op_lower),
        tel_list[non_io_idx] if non_io_idx < len(tel_list) else None,
      )
    non_io_idx += 1

    pred_shape, node_out_shape = _resolve_node_shapes(nid, graph, incoming_map, matched_tel, trace, default_in_shape)
    dtype_str = matched_tel.dtype if matched_tel is not None else "float32"
    ir_dt = _resolve_ir_dtype(dtype_str)

    _hydrate_ctor_and_linear_attrs(nid, node, op_lower, pred_shape, node_out_shape, trace)
    if "flatten" in op_lower and len(pred_shape) >= 2:
      node_out_shape = [pred_shape[0], _prod(pred_shape[1:])]
    _hydrate_spatial_conv_attrs(node, op_lower, pred_shape, node_out_shape)

    node.shape_metadata = tuple(node_out_shape)
    node.dtype = ir_dt
    node.output_specs = [TensorSpec(shape=tuple(node_out_shape), dtype=ir_dt)]
    node.attributes["num_elements"] = _prod(node_out_shape)
    if max_loop_trips is not None:
      node.attributes["loop_trip_count"] = max_loop_trips

    node.attributes["runtime_comment"] = _format_node_runtime_comment(
      output_shape=node_out_shape,
      dtype=dtype_str,
      grad_shape=matched_tel.grad_shape if matched_tel is not None else None,
      grad_norm=matched_tel.grad_norm if matched_tel is not None else None,
      dynamic_reason=matched_tel.dynamic_reason if matched_tel is not None else None,
    )


__all__ = ["_format_node_runtime_comment", "hydrate_graph_with_runtime_trace"]
