"""StableHLO Emitter Backend.

Translates Python CST to MLIR using the StableHLO dialect for math operations
and the Func/Builtin dialects for structure. It relies on the SemanticsManager
to map Python source APIs (like `torch.abs`) to StableHLO operations (like `stablehlo.abs`).
"""

import libcst as cst
from typing import List, Tuple, Optional, TYPE_CHECKING, Any, Union

from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.types import FunctionType
from ml_switcheroo.core.mlir.cst import (
  BlockNode,
  OperationNode,
  AttributeNode,
  RegionNode,
  TypeNode,
  ValueNode,
)
from ml_switcheroo.core.mlir.type_inference import parse_py_type_to_mlir, TypeInferencePass

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class StableHloEmitter(PythonToMlirEmitter):
  """Specialized Emitter that produces StableHLO, Func, and Builtin dialect operations."""

  def __init__(self, semantics: "SemanticsManager"):
    """Initialize the emitter with access to the Semantic Knowledge Base.

    Args:
        semantics: The manager instance to use for API resolution.

    """
    super().__init__()
    self.semantics = semantics

  def _emit_class_def(self, node: cst.ClassDef) -> OperationNode:
    """Maps Python Class to 'builtin.module'.

    Args:
        node: LibCST ClassDef node.

    Returns:
        MLIR OperationNode representing a module.

    """
    self.ctx.enter_scope()
    # Standard MLIR uses @name syntax for symbols, but module ops usually
    # take a symbol name attribute if nested, or just define a scope.
    # We model it as a nested module for structural parity.
    name_attr = AttributeNode(name="sym_name", value=f'"{node.name.value}"')
    attributes = [name_attr]

    region = RegionNode(blocks=[self._emit_block(node.body)])
    op = OperationNode(name="module", attributes=attributes, regions=[region])
    self.ctx.exit_scope()
    return op

  def _emit_func_def(self, node: cst.FunctionDef) -> OperationNode:
    """Maps Python Function to 'func.func'.

    Args:
        node: LibCST FunctionDef node.

    Returns:
        MLIR OperationNode representing the function.

    """
    self.ctx.enter_scope()
    func_name = node.name.value

    block_args = []
    initial_env = {}
    input_types = []

    for param in node.params.params:
      if isinstance(param.name, cst.Name):
        p_name = param.name.value
        val = self.ctx.allocate_ssa(prefix=f"%{p_name}")
        self.ctx.declare(p_name, val)

        # Type mapping
        mlir_type = parse_py_type_to_mlir("tensor<*xf32>")
        if param.annotation:
          anno_str = self._annotation_to_string(param.annotation.annotation)
          mlir_type = parse_py_type_to_mlir(anno_str)

        initial_env[p_name] = mlir_type
        input_types.append(mlir_type)
        block_args.append((val, TypeNode(body=mlir_type.to_string())))

    # Pre-pass type inference
    infer_pass = TypeInferencePass(initial_env=initial_env)
    node.body.visit(infer_pass)

    body_block = self._emit_block(node.body, label="^entry")
    body_block.arguments = block_args

    # Determine result types
    result_types = []
    mlir_res_types = []
    if node.returns:
      rt_str = self._annotation_to_string(node.returns.annotation)
      mlir_res_type = parse_py_type_to_mlir(rt_str)
      result_types.append(TypeNode(body=mlir_res_type.to_string()))
      mlir_res_types.append(mlir_res_type)
    elif infer_pass.return_types:
      for rt in infer_pass.return_types:
        result_types.append(TypeNode(body=rt.to_string()))
        mlir_res_types.append(rt)

    func_type = FunctionType(inputs=input_types, results=mlir_res_types)

    # FuncOp attributes
    attrs = [
      AttributeNode(name="sym_name", value=f'"{func_name}"'),
      AttributeNode(
        name="function_type",
        value=func_type.to_string(),
      ),
    ]

    op = OperationNode(
      name="func.func",
      attributes=attrs,
      regions=[RegionNode(blocks=[body_block])],
      result_types=result_types,
    )
    self.ctx.exit_scope()
    return op

  def _emit_while(self, node: cst.While) -> List[OperationNode]:
    """Maps Python While to 'stablehlo.while'.

    Args:
        node: LibCST While node.

    Returns:
        List of operations.
    """
    ops = []

    # 1. Condition Region
    cond_block = BlockNode(label="")
    cond_val, expr_ops = self._emit_expression(node.test)
    cond_block.operations.extend(expr_ops)
    cond_block.operations.append(OperationNode(name="stablehlo.return", operands=[cond_val]))
    cond_region = RegionNode(blocks=[cond_block])

    # 2. Body Region
    body_block = self._emit_block(node.body)
    if not body_block.operations:
      body_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))
    elif body_block.operations[-1].name not in ("func.return", "sw.return", "stablehlo.return"):
      body_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))
    body_region = RegionNode(blocks=[body_block])

    while_op = OperationNode(
      name="stablehlo.while",
      operands=[],  # We would need to capture state here
      regions=[cond_region, body_region],
      result_types=[],
    )
    ops.append(while_op)
    return ops

  def _emit_if(self, node: cst.If) -> List[OperationNode]:
    """Maps Python If to 'stablehlo.if' or 'stablehlo.case'.

    Currently handles basic if and else mapping to regions.

    Args:
        node: LibCST If node.

    Returns:
        List of operations.
    """
    ops = []
    # 1. Evaluate condition
    cond_val, expr_ops = self._emit_expression(node.test)
    ops.extend(expr_ops)

    # 2. Emit true branch region
    true_block = self._emit_block(node.body)

    # MLIR regions require a terminator. For now, we inject a dummy if none exists
    # A true implementation must track mutated state across branches.
    if not true_block.operations:
      true_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))
    elif true_block.operations[-1].name not in ("func.return", "sw.return", "stablehlo.return"):
      true_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))

    true_region = RegionNode(blocks=[true_block])
    regions = [true_region]

    # 3. Emit false branch region (if else exists)
    if getattr(node, "orelse", None):
      if isinstance(node.orelse, cst.Else):
        false_block = self._emit_block(node.orelse.body)
        if not false_block.operations:
          false_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))
        elif false_block.operations[-1].name not in ("func.return", "sw.return", "stablehlo.return"):
          false_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))
        false_region = RegionNode(blocks=[false_block])
        regions.append(false_region)
      else:  # isinstance(node.orelse, cst.If)
        # To be strictly compliant with stablehlo.if vs case, we handle elif as nested here
        false_block = BlockNode(label="", operations=self._emit_if(node.orelse))  # type: ignore
        if not false_block.operations:
          false_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))
        elif false_block.operations[-1].name not in ("func.return", "sw.return", "stablehlo.return"):  # pragma: no branch
          false_block.operations.append(OperationNode(name="stablehlo.return", operands=[]))
        false_region = RegionNode(blocks=[false_block])
        regions.append(false_region)
    else:
      # stablehlo.if requires two regions. If no else, emit empty region
      empty_block = BlockNode(label="", operations=[OperationNode(name="stablehlo.return", operands=[])])
      regions.append(RegionNode(blocks=[empty_block]))

    # We assume no return values (mutations) for this basic parity step.
    if_op = OperationNode(name="stablehlo.if", operands=[cond_val], regions=regions, result_types=[])
    ops.append(if_op)
    return ops

  def _emit_return(self, node: cst.Return) -> List[OperationNode]:
    """Maps Python Return to 'func.return'.

    Args:
        node: LibCST Return node.

    Returns:
        List of operations (expression evaluation + return).

    """
    ops = []
    operands = []
    if node.value:
      val, expr_ops = self._emit_expression(node.value)
      ops.extend(expr_ops)
      operands.append(val)

    op = OperationNode(name="func.return", operands=operands)
    ops.append(op)
    return ops

  def _emit_expression(self, expr: cst.BaseExpression) -> Tuple[ValueNode, List[OperationNode]]:
    """Overrides expression generation to intercept and resolve Semantic Operations.

    Args:
        expr: LibCST Expression node.

    Returns:
        Tuple of (Result Value, List of Ops).

    """
    if isinstance(expr, cst.Call):
      return self._emit_call(expr)

    # Delegate to base logic first
    val, ops = super()._emit_expression(expr)

    # Post-process the generated operations to resolve 'sw.op' to 'stablehlo.*'
    resolved_ops = []
    for op in ops:
      if op.name == "sw.op":
        self._resolve_sw_op(op)
      elif op.name == "sw.constant":
        self._resolve_sw_constant(op)
      resolved_ops.append(op)

    return val, resolved_ops

  def _emit_import(self, node: Union[cst.Import, cst.ImportFrom]) -> OperationNode:
    """Ignore imports in MLIR generation.

    Args:
        node: The import node.

    Returns:
        A dummy OperationNode that will be filtered out.
    """
    # StableHLO backend doesn't support 'sw.import', ignore it in generated MLIR
    # Return a dummy OperationNode that will be filtered out, or just pass
    return OperationNode(name="stablehlo.dummy_import", operands=[], attributes=[])

  def _emit_statement(self, stmt: cst.CSTNode) -> List[OperationNode]:
    """Emit a statement, filtering out dummy imports.

    Args:
        stmt: The statement node.

    Returns:
        A list of generated operations.
    """
    ops = super()._emit_statement(stmt)
    return [
      op for op in ops if getattr(op, "name", "") != "stablehlo.dummy_import" and getattr(op, "name", "") != "sw.import"
    ]

  def _resolve_sw_constant(self, op: OperationNode) -> None:
    """Mutates a 'sw.constant' node into a 'stablehlo.constant' node.

    Args:
        op: The operation node to mutate in-place.
    """
    op.name = "stablehlo.constant"
    val_attr = next((a for a in op.attributes if a.name == "value"), None)
    if val_attr:
      raw_val = val_attr.value
      # If string is quoted, unquote it
      if isinstance(raw_val, str) and raw_val.startswith('"') and raw_val.endswith('"'):
        raw_val = raw_val[1:-1]

      # Determine if it's a float or int
      is_float = "." in str(raw_val)
      mlir_type = "tensor<f32>" if is_float else "tensor<i32>"

      val_attr.value = f"dense<{raw_val}>"

      if not op.result_types:
        op.result_types = [TypeNode(body=mlir_type)]

  def _resolve_sw_op(self, op: OperationNode) -> None:
    """Mutates a 'sw.op' node into a 'stablehlo' node if a mapping exists.


    Removes the 'type' attribute upon successful resolution.

    Args:
        op: The operation node to mutate in-place.

    """
    # Find type attribute
    type_attr = next((a for a in op.attributes if a.name == "type"), None)
    if not type_attr:
      return

    api_name = str(type_attr.value).strip('"').strip("'")
    mapped_name = self._lookup_stablehlo_op(api_name)

    if mapped_name:
      op.name = mapped_name
      # Remove the 'type' attribute as it is now encoded in the op name
      op.attributes = [a for a in op.attributes if a.name != "type"]
      # Inject default tensor result type if missing
      if not op.result_types:
        op.result_types = [TypeNode(body="tensor<*xf32>")]

  def _lookup_stablehlo_op(self, api_name: str) -> Optional[str]:
    """Queries the SemanticsManager for the StableHLO variant of the given API.

    Args:
        api_name: Logic API string (e.g. 'torch.abs').

    Returns:
        StableHLO operation name (e.g. 'stablehlo.abs') or None.

    """
    # 1. Reverse lookup to get Abstract ID
    defn = self.semantics.get_definition(api_name)
    if not defn:
      return None

    _abstract_id, details = defn
    variants = details.get("variants", {})

    # 2. Check for 'stablehlo' variant
    if "stablehlo" in variants and variants["stablehlo"]:
      return variants["stablehlo"].get("api")  # type: ignore

    return None

  def _map_py_type_to_mlir(self, type_str: str) -> str:
    """Maps Python type strings to MLIR types.

    Args:
        type_str: Python Type Hint string.

    Returns:
        MLIR Type string (e.g. 'f32').

    """
    return parse_py_type_to_mlir(type_str).to_string()

  def _emit_call(self, expr: cst.Call) -> Tuple[ValueNode, List[OperationNode]]:
    """Handles semantic calls mapping them to StableHLO with specific attribute processing.

    Args:
        expr: Call expression.

    Returns:
        Tuple of Result Value, List of Ops.
    """
    flat_name = self._flatten_attr(expr.func)
    stablehlo_name = None
    if flat_name:
      stablehlo_name = self._lookup_stablehlo_op(flat_name)

    if not stablehlo_name:
      # Fall back to default logic
      val, ops = super()._emit_expression(expr)
      resolved_ops = []
      for op in ops:
        if op.name == "sw.op":
          self._resolve_sw_op(op)
        elif op.name == "sw.constant":
          self._resolve_sw_constant(op)
        resolved_ops.append(op)
      return val, resolved_ops

    # Process as StableHLO operation
    ops = []
    operands = []
    attributes = []
    regions = []

    # Map kwargs to attributes and positional args to operands
    for arg in expr.args:
      if arg.keyword:
        kw = arg.keyword.value
        attr_val = self._extract_literal(arg.value)
        attributes.append(AttributeNode(name=kw, value=attr_val))
      else:
        # Handle lambdas or functions for higher-order ops like reduce

        if isinstance(arg.value, cst.Lambda):
          self.ctx.enter_scope()
          # Emit Lambda body into a region
          lambda_block = BlockNode(label="^bb0")
          # Basic assumption of lambda args matching reduction signature
          l_args = []
          for p in arg.value.params.params:
            pname = p.name.value
            pval = self.ctx.allocate_ssa(prefix=f"%{pname}")
            self.ctx.declare(pname, pval)
            l_args.append((pval, TypeNode(body="tensor<*xf32>")))

          lambda_block.arguments = l_args

          # Emit the expression inside the lambda
          res_val, expr_ops = self._emit_expression(arg.value.body)
          lambda_block.operations.extend(expr_ops)
          lambda_block.operations.append(OperationNode(name="stablehlo.return", operands=[res_val]))
          regions.append(RegionNode(blocks=[lambda_block]))
          self.ctx.exit_scope()
        else:
          v, o = self._emit_expression(arg.value)
          ops.extend(o)
          operands.append(v)

    # Add any StableHLO-specific formatting (e.g., dense element attributes)
    processed_attrs = []
    for attr in attributes:
      if isinstance(attr.value, list):
        # E.g., padding=[1, 1] -> dense<[1, 1]>
        val_str = f"dense<[{', '.join(str(x) for x in attr.value)}]>"
        processed_attrs.append(AttributeNode(name=attr.name, value=val_str, type_annotation="tensor<2xi64>"))
      elif isinstance(attr.value, str):
        processed_attrs.append(AttributeNode(name=attr.name, value=f'"{attr.value}"'))
      else:
        processed_attrs.append(AttributeNode(name=attr.name, value=str(attr.value)))

    result = self.ctx.allocate_ssa()
    op = OperationNode(
      name=stablehlo_name,
      results=[result],
      operands=operands,
      attributes=processed_attrs,
      regions=regions,
      result_types=[TypeNode(body="tensor<*xf32>")],  # Default return, needs refinement if needed
    )
    ops.append(op)
    return result, ops

  def _extract_literal(self, node: cst.CSTNode) -> Any:
    """Extracts python literal from CST node.

    Args:
        node: The CST node containing a literal value.

    Returns:
        The extracted Python literal value.
    """
    if isinstance(node, cst.Integer):
      return int(node.value)
    elif isinstance(node, cst.Float):
      return float(node.value)
    elif isinstance(node, cst.SimpleString):
      return node.value.strip("\"'")
    elif isinstance(node, (cst.List, cst.Tuple)):
      return [self._extract_literal(el.value) for el in node.elements]
    return "%error"
