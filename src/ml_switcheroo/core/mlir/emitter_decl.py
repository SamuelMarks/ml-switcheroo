"""MLIR SSA Context and Declaration Emitter Delegate.

Provides ``SSAContext`` (managing lexical SSA symbol tables and ID allocation)
and ``MlirDeclEmitter`` (converting Python ``ClassDef`` and ``FunctionDef`` CST
nodes into ``sw.module`` and ``sw.func`` MLIR operations).
"""

import typing

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, OperationNode, RegionNode, TypeNode, ValueNode
from ml_switcheroo.core.mlir.types import format_mlir_tensor_type
from ml_switcheroo.core.runtime_models import RuntimeTraceReport


class SSAContext:
  """Manage Single Static Assignment (SSA) variable scopes and ID allocation."""

  def __init__(self) -> None:
    """Initialize the context with a root scope."""
    self._scopes: list[dict[str, ValueNode]] = [{}]
    self._counter: int = 0

  def enter_scope(self) -> None:
    """Push a new variable scope onto the stack."""
    self._scopes.append({})

  def exit_scope(self) -> None:
    """Pop the current variable scope from the stack."""
    if len(self._scopes) > 1:
      self._scopes.pop()

  def declare(self, name: str, value: ValueNode) -> None:
    """Register a variable name in the current scope.

    Args:
        name: Python variable identifier.
        value: MLIR ``ValueNode`` (SSA value) associated with ``name``.
    """
    self._scopes[-1][name] = value

  def lookup(self, name: str) -> ValueNode | None:
    """Resolve a Python variable name to its current SSA value from innermost to outermost scope.

    Args:
        name: Python identifier to look up.

    Returns:
        Associated ``ValueNode`` or ``None`` if unbound.
    """
    for scope in reversed(self._scopes):
      if name in scope:
        return scope[name]
    return None

  def allocate_ssa(self, prefix: str = "%") -> ValueNode:
    """Generate a new unique SSA ``ValueNode``.

    Args:
        prefix: String prefix for the SSA identifier (defaults to ``'%'``).

    Returns:
        Newly allocated ``ValueNode`` (e.g. ``%0``, ``%1``).
    """
    val = ValueNode(name=f"{prefix}{self._counter}")
    self._counter += 1
    return val


class MlirDeclEmitterHost(typing.Protocol):
  """Structural protocol required by ``MlirDeclEmitter``.

  Attributes:
      ctx: Active ``SSAContext`` instance.
      runtime_trace: Optional ``RuntimeTraceReport`` for shape telemetry.
  """

  ctx: SSAContext
  runtime_trace: RuntimeTraceReport | None

  def _flatten_attr(self, node: cst.CSTNode) -> str | None:
    """Flatten a ``Name`` or ``Attribute`` chain into a dotted string.

    Args:
        node: CST node to flatten.

    Returns:
        Dotted string or ``None``.
    """
    ...

  def _emit_block(self, body_enc: cst.BaseSuite | typing.Sequence[cst.CSTNode], label: str = "") -> BlockNode:
    """Emit a ``BlockNode`` from a CST suite or statement sequence.

    Args:
        body_enc: CST suite or statement sequence.
        label: Optional block label.

    Returns:
        Emitted ``BlockNode``.
    """
    ...

  def _annotation_to_string(self, node: cst.CSTNode) -> str:
    """Flatten a CST type annotation node into a string.

    Args:
        node: CST type annotation node.

    Returns:
        Flattened annotation string.
    """
    ...


class MlirDeclEmitter:
  """Single-responsibility delegate lowering Python class and function declarations to MLIR.

  Attributes:
      host: Bound ``MlirDeclEmitterHost`` instance.
  """

  def __init__(self, host: MlirDeclEmitterHost) -> None:
    """Initialize ``MlirDeclEmitter`` bound to ``host``.

    Args:
        host: Bound ``MlirDeclEmitterHost`` instance.
    """
    self.host: MlirDeclEmitterHost = host

  def emit_class_def(self, node: cst.ClassDef) -> OperationNode:
    """Convert a Python ``cst.ClassDef`` into an ``sw.module`` ``OperationNode``.

    Args:
        node: ``cst.ClassDef`` node to lower.

    Returns:
        Constructed ``sw.module`` ``OperationNode``.
    """
    self.host.ctx.enter_scope()
    attributes: list[AttributeNode] = [AttributeNode(name="sym_name", value=f'"{node.name.value}"')]

    if node.bases:
      base_names: list[str] = []
      for b in node.bases:
        flat_name = self.host._flatten_attr(b.value)
        if flat_name:
          base_names.append(f'"{flat_name}"')
      if base_names:
        attributes.append(AttributeNode(name="bases", value=base_names))

    region = RegionNode(blocks=[self.host._emit_block(node.body)])
    op = OperationNode(name="sw.module", attributes=attributes, regions=[region])
    self.host.ctx.exit_scope()
    return op

  def emit_func_def(self, node: cst.FunctionDef) -> OperationNode:
    """Convert a Python ``cst.FunctionDef`` into an ``sw.func`` ``OperationNode``.

    Args:
        node: ``cst.FunctionDef`` node to lower.

    Returns:
        Constructed ``sw.func`` ``OperationNode`` with block arguments.
    """
    self.host.ctx.enter_scope()
    func_name = node.name.value
    block_args: list[tuple[ValueNode, TypeNode]] = []
    trace = self.host.runtime_trace
    is_inf = func_name not in ("__init__", "setup")
    in_idx = 0

    for param in node.params.params:
      if isinstance(param.name, cst.Name):
        p_name = param.name.value
        val = self.host.ctx.allocate_ssa(prefix=f"%{p_name}")
        self.host.ctx.declare(p_name, val)
        t_str = "!sw.unknown"
        if param.annotation:
          t_str = f'!sw.type<"{self.host._annotation_to_string(param.annotation.annotation)}">'
        elif (
          is_inf and p_name not in ("self", "cls", "rngs") and trace is not None and trace.executed and trace.input_shapes
        ):
          shape_idx = min(in_idx, len(trace.input_shapes) - 1)
          t_str = format_mlir_tensor_type(trace.input_shapes[shape_idx], "float32")
          in_idx += 1
        block_args.append((val, TypeNode(body=t_str)))

    body_block = self.host._emit_block(node.body, label="^entry")
    body_block.arguments = block_args
    op = OperationNode(
      name="sw.func",
      attributes=[AttributeNode(name="sym_name", value=f'"{func_name}"')],
      regions=[RegionNode(blocks=[body_block])],
    )
    self.host.ctx.exit_scope()
    return op


__all__ = ["MlirDeclEmitter", "MlirDeclEmitterHost", "SSAContext"]
