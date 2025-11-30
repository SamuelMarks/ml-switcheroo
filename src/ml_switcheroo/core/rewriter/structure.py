"""
Structural Rewriting Logic.

This module provides the `StructureMixin`, a component of the PivotRewriter
responsible for transforming the scaffolding of Python code: Class definitions
and Function definitions.

It implements logic for:
1.  **Inheritance Warping**: Swapping framework base classes (e.g., `torch.nn.Module`
    ↔ `flax.nnx.Module`).
2.  **Signature Transformation**: Injecting or stripping framework-specific
    state arguments (e.g., `rngs`) in constructors.
3.  **Method Renaming**: Mapping `forward` ↔ `__call__`.
4.  **Constructor Logic Injection**: Ensuring `super().__init__()` is present
    when targeting PyTorch.
"""

from typing import Optional
import libcst as cst

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.types import SignatureContext


class StructureMixin(BaseRewriter):
  """
  Mixin for transforming structural elements (Classes, Functions).

  Attributes:
      _in_module_class (bool): Inherited from BaseRewriter. True if currently
          traversing a Neural Network Module class.
      _signature_stack (List[SignatureContext]): Inherited from BaseRewriter.
          Tracks current function signature state.
  """

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """
    Visits a class definition to detect Neural Module context.

    It inspects base classes to determine if the class is a framework-specific
    Neural Module (Torch or Flax NNX). This sets the `_in_module_class` flag,
    enabling method renaming and signature fixups nested within.

    Args:
        node: The libCST ClassDef node.

    Returns:
        True to traverse children.
    """
    self._enter_scope()

    is_module = False
    for base in node.bases:
      name = self._get_qualified_name(base.value)
      if not name:
        continue

      # Robust PyTorch check
      if "Module" in name and (
        "torch.nn" in name or name.startswith("nn.") or name == "nn.Module" or name == "torch.nn.Module"
      ):
        is_module = True
        break

      # Robust Flax check
      if "Module" in name and ("flax.nnx" in name or "nnx" in name):
        is_module = True
        break

    if is_module:
      self._in_module_class = True

    return True

  def leave_ClassDef(self, _original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    """
    Rewrites class inheritance for the target framework.

    If the class was identified as a Neural Module, this method swaps the
    base class:
    - Target JAX: `torch.nn.Module` → `flax.nnx.Module`
    - Target Torch: `flax.nnx.Module` → `torch.nn.Module`

    Args:
        original_node: The original CST node (unused).
        updated_node: The CST node with transformed children.

    Returns:
        The transformed ClassDef node.
    """
    self._exit_scope()

    if self._in_module_class:
      self._in_module_class = False  # Reset flag

      # Rewrite Parent Class
      new_bases = []
      for base in updated_node.bases:
        name = self._get_qualified_name(base.value)
        # Ensure we only replace the Module inheritance, preserving mixins
        if name and "Module" in name:
          if self.target_fw == "jax":
            new_base = cst.Arg(value=self._create_dotted_name("flax.nnx.Module"))
            new_bases.append(new_base)
            continue
          elif self.target_fw == "torch":
            new_base = cst.Arg(value=self._create_dotted_name("torch.nn.Module"))
            new_bases.append(new_base)
            continue

        new_bases.append(base)

      updated_node = updated_node.with_changes(bases=new_bases)

    return updated_node

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """
    Visits function definitions to track signature context.

    Captures existing arguments to allow smart injection/stripping later.
    Pushes a new `SignatureContext` onto the stack.

    Args:
        node: The CST FunctionDef node.

    Returns:
        True to traverse children.
    """
    self._enter_scope()

    existing_args = set()
    for param in node.params.params:
      if isinstance(param.name, cst.Name):
        existing_args.add(param.name.value)

    is_init = node.name.value == "__init__"

    self._signature_stack.append(
      SignatureContext(
        existing_args=existing_args,
        is_init=is_init,
        is_module_method=self._in_module_class,
      )
    )
    return True

  def leave_FunctionDef(self, _original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
    """
    Finalizes function transformations.

    Applies:
    1.  **Method Renaming**: `forward` ↔ `__call__`.
    2.  **Signature Modification**: Injecting/Stripping `rngs` for Flax NNX support.
    3.  **Plugin Injection**: Applying args requested by plugins.
    4.  **Preamble Injection**: Inserting setup code (like `super().__init__()` or RNG split).

    Args:
        original_node: The original CST node.
        updated_node: The CST node with transformed children.

    Returns:
        The transformed FunctionDef node.
    """
    self._exit_scope()

    if not self._signature_stack:
      return updated_node

    sig_ctx = self._signature_stack.pop()

    # 1. Rename Methods
    if sig_ctx.is_module_method:
      if self.target_fw == "jax":
        # PyTorch -> JAX: forward -> __call__
        if updated_node.name.value == "forward":
          updated_node = updated_node.with_changes(name=cst.Name("__call__"))
      elif self.target_fw == "torch":
        # JAX -> PyTorch: __call__ -> forward
        if updated_node.name.value == "__call__":
          updated_node = updated_node.with_changes(name=cst.Name("forward"))

    # 2. Modify __init__ signature & body
    if sig_ctx.is_init and sig_ctx.is_module_method:
      # JAX: Inject 'rngs'
      if self.target_fw == "jax":
        if "rngs" not in sig_ctx.existing_args:
          found = any(n == "rngs" for n, _ in sig_ctx.injected_args)
          if not found:
            sig_ctx.injected_args.append(("rngs", "flax.nnx.Rngs"))

      # Torch: Strip 'rngs' and ensure super().__init__()
      elif self.target_fw == "torch":
        updated_node = self._strip_argument_from_signature(updated_node, "rngs")
        updated_node = self._ensure_super_init(updated_node)

    # 3. Apply Injected Arguments (General Plugins)
    for name, annotation in sig_ctx.injected_args:
      updated_node = self._inject_argument_to_signature(updated_node, name, annotation)

    # 4. Apply Preamble Statements
    if sig_ctx.preamble_stmts:
      new_stmts = []
      for stmt_code in sig_ctx.preamble_stmts:
        try:
          parsed_stmt = cst.parse_statement(stmt_code)
          new_stmts.append(parsed_stmt)
        except cst.ParserSyntaxError:
          self._report_failure(f"Failed to inject preamble: {stmt_code}")

      original_body_stmts = list(updated_node.body.body)
      # Find safe insertion point (after docstring)
      insert_idx = 0
      if (
        original_body_stmts
        and isinstance(original_body_stmts[0], cst.SimpleStatementLine)
        and isinstance(original_body_stmts[0].body[0], cst.Expr)
        and isinstance(
          original_body_stmts[0].body[0].value,
          (cst.SimpleString, cst.ConcatenatedString),
        )
      ):
        insert_idx = 1

      updated_body = updated_node.body.with_changes(
        body=original_body_stmts[:insert_idx] + new_stmts + original_body_stmts[insert_idx:]
      )
      updated_node = updated_node.with_changes(body=updated_body)

    return updated_node

  def _inject_argument_to_signature(
    self, node: cst.FunctionDef, arg_name: str, annotation: Optional[str] = None
  ) -> cst.FunctionDef:
    """
    Helper to append a new argument to the function definition.

    Inserts the new parameter after `self` if present, otherwise at index 0.

    Args:
        node: The function definition node to modify.
        arg_name: The name of the argument to inject.
        annotation: Optional type hint string (e.g., 'jax.Array').

    Returns:
        The modified FunctionDef node.
    """
    params = list(node.params.params)

    # Determine insertion index (after self)
    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    anno_node = None
    if annotation:
      anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation))

    new_param = cst.Param(
      name=cst.Name(arg_name),
      annotation=anno_node,
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )

    if 0 < insert_idx <= len(params):
      prev_param = params[insert_idx - 1]
      if prev_param.comma == cst.MaybeSentinel.DEFAULT:
        params[insert_idx - 1] = prev_param.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    params.insert(insert_idx, new_param)
    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)

  def _strip_argument_from_signature(self, node: cst.FunctionDef, arg_name: str) -> cst.FunctionDef:
    """
    Helper to remove an argument from the function signature.

    Args:
        node: The function definition node.
        arg_name: Name of the argument to remove (e.g. 'rngs').

    Returns:
        The modified function def.
    """
    params = list(node.params.params)
    new_params = [p for p in params if not (isinstance(p.name, cst.Name) and p.name.value == arg_name)]

    # Clean trailing commas on the new last element if needed
    if new_params and new_params[-1].comma != cst.MaybeSentinel.DEFAULT:
      last = new_params[-1]
      new_params[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=new_params)
    return node.with_changes(params=new_params_node)

  def _ensure_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """
    Injects `super().__init__()` into `__init__` if missing.

    Required for valid PyTorch sub-classing. Validates if the call
    already exists to avoid duplication. Wraps insertion logic to respect
    docstrings.

    Args:
        node: The `__init__` function definition node.

    Returns:
        The modified node with the super call injected.
    """
    # 1. Check if super().__init__ already exists
    for stmt in node.body.body:
      if isinstance(stmt, cst.SimpleStatementLine):
        for small in stmt.body:
          if isinstance(small, cst.Expr) and isinstance(small.value, cst.Call):
            # Detect super().__init__ or super(CLS, self).__init__
            call = small.value
            if isinstance(call.func, cst.Attribute) and call.func.attr.value == "__init__":
              # Check if receiver is super()
              receiver = call.func.value
              if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
                if receiver.func.value == "super":
                  return node

    # 2. Construct the statement
    super_stmt = cst.SimpleStatementLine(
      body=[
        cst.Expr(
          value=cst.Call(
            func=cst.Attribute(
              value=cst.Call(func=cst.Name("super")),
              attr=cst.Name("__init__"),
            )
          )
        )
      ]
    )

    # 3. Insert in body
    stmts = list(node.body.body)
    insert_idx = 0

    # Skip docstring
    if (
      stmts
      and isinstance(stmts[0], cst.SimpleStatementLine)
      and isinstance(stmts[0].body[0], cst.Expr)
      and isinstance(stmts[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    ):
      insert_idx = 1

    stmts.insert(insert_idx, super_stmt)
    return node.with_changes(body=node.body.with_changes(body=stmts))
