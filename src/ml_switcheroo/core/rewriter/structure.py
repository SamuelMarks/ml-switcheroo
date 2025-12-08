"""
Structural Rewriting Logic.

This module provides the `StructureMixin`, a component of the PivotRewriter
responsible for transforming the scaffolding of Python code: Class definitions
and Function definitions.

It implements logic for:
1.  **Inheritance Warping**: Swapping framework base classes (e.g., `torch.nn.Module`
    ↔ `flax.nnx.Module` ↔ `keras.Layer`).
2.  **Signature Transformation**: Injecting or stripping framework-specific
    state arguments (e.g., `rngs`) in constructors.
3.  **Method Renaming**: Mapping `forward` ↔ `__call__` ↔ `call`.
4.  **Constructor Logic Injection**: Ensuring `super().__init__()` is present
    when targeting PyTorch or Keras.
5.  **Type Hint Rewriting**: Mapping type annotations (e.g., `torch.Tensor` -> `jax.Array`)
    in function signatures or variable assignments.
6.  **Docstring Updates**: Automatically documenting injected arguments.
"""

from typing import Optional, List, Tuple
import libcst as cst
import re

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.types import SignatureContext


class StructureMixin(BaseRewriter):
  """
  Mixin for transforming structural elements (Classes, Functions) and Type Hints.

  Attributes:
      _in_module_class (bool): Inherited from BaseRewriter. True if currently
          traversing a Neural Network Module class.
      _signature_stack (List[SignatureContext]): Inherited from BaseRewriter.
          Tracks current function signature state.
      _in_annotation (bool): Tracks if the visitor is currently inside a type annotation.
  """

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """
    Visits a class definition to detect Neural Module context.

    It inspects base classes to determine if the class is a framework-specific
    Neural Module (Torch, Flax NNX, or Keras). This sets the `_in_module_class` flag,
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

      if self._is_framework_base(name):
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
    - Target JAX: `...` → `flax.nnx.Module`
    - Target Torch: `...` → `torch.nn.Module`
    - Target Keras/TF: `...` → `keras.Layer`

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

        # Ensure we only replace the Module/Layer inheritance, preserving mixins
        if self._is_framework_base(name):
          target_base = None

          if self.target_fw == "jax":
            target_base = "flax.nnx.Module"
          elif self.target_fw == "torch":
            target_base = "torch.nn.Module"
          elif self.target_fw in ["tensorflow", "keras"]:
            target_base = "keras.Layer"

          if target_base:
            new_base_node = cst.Arg(value=self._create_dotted_name(target_base))
            new_bases.append(new_base_node)
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
    1.  **Method Renaming**: `forward` ↔ `__call__` ↔ `call`.
    2.  **Signature Modification**: Injecting/Stripping `rngs` for Flax NNX support.
    3.  **Plugin Injection**: Applying args requested by plugins.
    4.  **Preamble Injection**: Inserting setup code (like `super().__init__()` or RNG split).
    5.  **Docstring Updates**: Appends injected arguments to docstrings.

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
      curr_name = updated_node.name.value
      target_name = None

      if self.target_fw == "jax":
        target_name = "__call__"
      elif self.target_fw == "torch":
        target_name = "forward"
      elif self.target_fw in ["tensorflow", "keras"]:
        target_name = "call"

      # Rename if current is a standard forward-pass method (forward, __call__, call)
      # and it doesn't match target convention
      known_methods = {"forward", "__call__", "call"}
      if target_name and curr_name in known_methods and curr_name != target_name:
        updated_node = updated_node.with_changes(name=cst.Name(target_name))

    # 2. Modify __init__ signature & body
    if sig_ctx.is_init and sig_ctx.is_module_method:
      # JAX: Inject 'rngs'
      if self.target_fw == "jax":
        if "rngs" not in sig_ctx.existing_args:
          found = any(n == "rngs" for n, _ in sig_ctx.injected_args)
          if not found:
            sig_ctx.injected_args.append(("rngs", "flax.nnx.Rngs"))

      # Torch / Keras: Strip 'rngs' and ensure super().__init__()
      elif self.target_fw in ["torch", "tensorflow", "keras"]:
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

    # 5. Update Docstring with Injected Arguments
    if sig_ctx.injected_args:
      updated_node = self._update_docstring(updated_node, sig_ctx.injected_args)

    return updated_node

  def visit_Annotation(self, node: cst.Annotation) -> Optional[bool]:
    """
    Enters a type annotation node (e.g., `: torch.Tensor` or `-> int`).
    Sets a flag to allow `leave_Name` to rewrite type names.
    """
    self._in_annotation = True
    return True

  def leave_Annotation(self, original_node: cst.Annotation, updated_node: cst.Annotation) -> cst.Annotation:
    """
    Leaves a type annotation node. Resets the annotation flag.
    """
    self._in_annotation = False
    return updated_node

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    """
    Rewrites Names found within Type Annotations.

    If we are inside an annotation context (e.g., `x: Tensor`), we resolve
    this name via aliases (e.g., `Tensor` -> `torch.Tensor`), look it up
    in the semantics, and rewrite it if a mapping exists (e.g., -> `jax.Array`).

    Note: Calls and Attributes (e.g. `torch.Tensor`) are handled by other mixins
    or `leave_Attribute` regardless of context, but bare `Name` nodes in code
    are usually variables we don't want to touch. This method is scoped strictly
    to annotations to be safe.
    """
    if getattr(self, "_in_annotation", False):
      full_name = self._get_qualified_name(original_node)
      if full_name:
        mapping = self._get_mapping(full_name)
        if mapping and "api" in mapping:
          return self._create_name_node(mapping["api"])

    return updated_node

  def _is_framework_base(self, name: str) -> bool:
    """
    Checks if the given class name corresponds to a known Framework Module class.
    """
    if not name:
      return False

    # PyTorch
    if name == "torch.nn.Module" or name == "nn.Module":
      return True

    # Flax NNX
    if name == "flax.nnx.Module" or name == "nnx.Module":
      return True

    # Keras (including TF Keras)
    if ("keras" in name or "tf" in name) and ("Layer" in name or "Model" in name):
      return True

    return False

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

    # Clean trailing commas on new last element if needed
    if new_params and new_params[-1].comma != cst.MaybeSentinel.DEFAULT:
      last = new_params[-1]
      new_params[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=new_params)
    return node.with_changes(params=new_params_node)

  def _ensure_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """
    Injects `super().__init__()` into `__init__` if missing.

    Required for valid PyTorch and Keras sub-classing. Validates if the call
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

  def _update_docstring(self, node: cst.FunctionDef, injected_args: List[Tuple[str, Optional[str]]]) -> cst.FunctionDef:
    """
    Updates the function docstring to include injected arguments.

    It parses the existing docstring (handling Google/NumPy style if possible)
    and appends a new entry for arguments like `rngs` or `key`.

    Args:
        node: The function definition node.
        injected_args: List of (name, annotation) injected.

    Returns:
        The function definition with updated docstring.
    """
    body = node.body
    if not body.body:
      return node

    # Check first statement for docstring
    stmt = body.body[0]
    if not isinstance(stmt, cst.SimpleStatementLine):
      return node
    if len(stmt.body) != 1 or not isinstance(stmt.body[0], cst.Expr):
      return node

    expr_node = stmt.body[0].value
    if not isinstance(expr_node, cst.SimpleString):
      # Concatenated strings are complex to edit, skipping
      return node

    raw_val = expr_node.value

    # Parse quote style
    quote_style = '"""'
    prefix = ""

    if raw_val.startswith(("r", "u", "R", "U")):
      prefix = raw_val[0]
      content_start = raw_val[1:]
    else:
      content_start = raw_val

    if content_start.startswith('"""'):
      quote_style = '"""'
    elif content_start.startswith("'''"):
      quote_style = "'''"
    elif content_start.startswith('"'):
      quote_style = '"'
    elif content_start.startswith("'"):
      quote_style = "'"
    else:
      return node  # Unknown format

    # Extract inner content
    # Remove quotes from start and end
    # Note: len(quote_style) could be 1 or 3
    q_len = len(quote_style)
    if len(content_start) < 2 * q_len:
      return node

    inner_text = content_start[q_len:-q_len]

    # Modify
    new_text = self._modify_docstring_text(inner_text, injected_args)

    if new_text == inner_text:
      return node

    # Reconstruct
    new_val = f"{prefix}{quote_style}{new_text}{quote_style}"
    new_expr = expr_node.with_changes(value=new_val)
    new_stmt = stmt.with_changes(body=[stmt.body[0].with_changes(value=new_expr)])
    new_body_stmts = list(body.body)
    new_body_stmts[0] = new_stmt

    return node.with_changes(body=body.with_changes(body=new_body_stmts))

  def _modify_docstring_text(self, text: str, args: List[Tuple[str, Optional[str]]]) -> str:
    """
    Heuristic text modification.
    Support Google (Args:) and NumPy (Parameters) section extraction/appending.
    """
    # Detect Indentation
    lines = text.splitlines()
    if not lines:
      return text  # Empty docstring?

    # Find indentation of the prompt block (usually second line determines indentation)
    indent = ""
    for line in lines[1:]:
      if line.strip():
        indent = re.match(r"^\s*", line).group(0)
        break

    if not indent and len(lines) > 0:
      # Fallback for single line or no indentation detected
      indent = "    "

    # Prepare Entries
    new_entries = []
    for name, _ in args:
      # Skip if already present
      if re.search(rf"\b{name}\b\s*[:\(]", text) or re.search(rf"\b{name}\s+:", text):
        continue

      entry = f"{indent}{name}: Injected state argument."
      new_entries.append(entry)

    if not new_entries:
      return text

    # Logic:
    # 1. Look for 'Args:'
    # 2. Look for 'Parameters' (NumPy style often underlined)
    # 3. Else Append 'Args:' section

    # 1. Google Style 'Args:'
    args_match = re.search(r"(\n\s*Args:\s*\n)", text)
    if args_match:
      split_idx = args_match.end()
      block = "\n".join(new_entries)
      return text[:split_idx] + block + "\n" + text[split_idx:]

    # 2. NumPy Style 'Parameters\n--------'
    param_match = re.search(r"(\n\s*Parameters\s*\n\s*[-=]+\s*\n)", text)
    if param_match:
      split_idx = param_match.end()
      block = "\n".join(new_entries)
      return text[:split_idx] + block + "\n" + text[split_idx:]

    # 3. Append to end (if multiline)
    if len(lines) > 0:
      # Try to insert before the closing quotes (which are not here, it's inner text)
      # We just append at appropriate collected indentation
      final_indent = re.match(r"^\s*", lines[-1]).group(0) if lines[-1].strip() == "" else indent

      header = f"\n\n{final_indent}Args:\n"
      block = "\n".join([f"{final_indent}    {e.strip()}" for e in new_entries])
      return text + header + block + "\n"

    return text
