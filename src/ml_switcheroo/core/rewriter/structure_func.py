"""
Function Structure Rewriting Logic.

Handles transformations relative to function definitions, specifically:
1.  **Logic 5: Method Renaming**: Mapping `forward` <-> `__call__` <-> `call` using Configuration Traits.
    *Decoupling Update*: Uses `known_inference_methods` set from traits to detect candidate methods,
    allowing custom frameworks (like `fastai.predict`) to be recognized without editing core code.
2.  Signature Modification: Injecting hooks or state arguments (Logic 2).
3.  Body Injection: Preamble handling (super init, rng splitting).
4.  Docstring Updating.
"""

import re
from typing import Optional, List, Tuple, Set
import libcst as cst
from libcst import BaseSuite, SimpleStatementSuite, IndentedBlock, SimpleStatementLine

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.core.escape_hatch import EscapeHatch


class FuncStructureMixin(BaseRewriter):
  """
  Mixin for transforming FunctionDef nodes via Traits.
  """

  def _get_traits(self) -> StructuralTraits:
    try:
      # Access traits for the TARGET framework
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "traits" in config_dict:
          return StructuralTraits.model_validate(config_dict["traits"])
    except Exception:
      pass
    return StructuralTraits()

  def _get_source_inference_methods(self) -> Set[str]:
    """
    Lazily loads the set of known inference methods for the Source Framework.
    This allows flexible detection (e.g. 'predict' vs 'forward') without hardcoding.
    """
    default_methods = {"forward", "__call__", "call"}

    try:
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.source_fw)
        if config_dict and "traits" in config_dict:
          traits = StructuralTraits.model_validate(config_dict["traits"])
          if traits.known_inference_methods:
            return traits.known_inference_methods
    except Exception:
      pass

    return default_methods

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
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
    self._exit_scope()

    if not self._signature_stack:
      return updated_node

    sig_ctx = self._signature_stack.pop()
    target_traits = self._get_traits()

    if sig_ctx.is_module_method:
      curr_name = updated_node.name.value
      target_name = target_traits.forward_method

      # DECOUPLING FIX: Use traits from Source Framework (or shared config) to detect candidates
      known_methods = self._get_source_inference_methods()

      # If current method is a known inference method (e.g. 'forward'), rename it to target (e.g. '__call__')
      if target_name and curr_name in known_methods and curr_name != target_name:
        updated_node = updated_node.with_changes(name=cst.Name(target_name))

      if sig_ctx.is_init and target_traits.init_method_name and target_traits.init_method_name != "__init__":
        updated_node = updated_node.with_changes(name=cst.Name(target_traits.init_method_name))

    if sig_ctx.is_init and sig_ctx.is_module_method:
      for arg_name, arg_type in target_traits.inject_magic_args:
        if arg_name not in sig_ctx.existing_args:
          found = any(n == arg_name for n, _ in sig_ctx.injected_args)
          if not found:
            sig_ctx.injected_args.append((arg_name, arg_type))

      for arg_name in target_traits.strip_magic_args:
        updated_node = self._strip_argument_from_signature(updated_node, arg_name)

      if target_traits.requires_super_init:
        updated_node = self._ensure_super_init(updated_node)
      else:
        # Logic 3: Strip super() if not required
        updated_node = self._strip_super_init(updated_node)

    for name, annotation in sig_ctx.injected_args:
      updated_node = self._inject_argument_to_signature(updated_node, name, annotation)

    if sig_ctx.preamble_stmts:
      updated_node = self._apply_preamble(updated_node, sig_ctx.preamble_stmts)

    if sig_ctx.injected_args:
      updated_node = self._update_docstring(updated_node, sig_ctx.injected_args)

    return updated_node

  def _is_body_accessible(self, body: BaseSuite) -> bool:
    return isinstance(body, IndentedBlock)

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    if isinstance(node.body, SimpleStatementSuite):
      new_body_stmts = []
      for stmt in node.body.body:
        new_body_stmts.append(cst.SimpleStatementLine(body=[stmt]))

      new_block = cst.IndentedBlock(body=new_body_stmts)
      return node.with_changes(body=new_block)
    return node

  def _inject_argument_to_signature(
    self, node: cst.FunctionDef, arg_name: str, annotation: Optional[str] = None
  ) -> cst.FunctionDef:
    params = list(node.params.params)

    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    anno_node = None
    if annotation:
      anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation))

    new_param = cst.Param(name=cst.Name(arg_name), annotation=anno_node, comma=cst.MaybeSentinel.DEFAULT)
    params.insert(insert_idx, new_param)

    for i in range(len(params) - 1):
      if params[i].comma == cst.MaybeSentinel.DEFAULT:
        params[i] = params[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if len(params) > 0:
      last = params[-1]
      if last.comma != cst.MaybeSentinel.DEFAULT:
        params[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)

  def _strip_argument_from_signature(self, node: cst.FunctionDef, arg_name: str) -> cst.FunctionDef:
    params = list(node.params.params)
    new_params = [p for p in params if not (isinstance(p.name, cst.Name) and p.name.value == arg_name)]

    for i in range(len(new_params) - 1):
      if new_params[i].comma == cst.MaybeSentinel.DEFAULT:
        new_params[i] = new_params[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if len(new_params) > 0:
      new_params[-1] = new_params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=new_params)
    return node.with_changes(params=new_params_node)

  def _ensure_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    if isinstance(node.body, SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    if self._has_super_init(node):
      return node

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

    stmts = list(node.body.body)
    insert_idx = 0
    if (
      stmts
      and isinstance(stmts[0], cst.SimpleStatementLine)
      and isinstance(stmts[0].body[0], cst.Expr)
      and isinstance(stmts[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    ):
      insert_idx = 1

    stmts.insert(insert_idx, super_stmt)
    return node.with_changes(body=node.body.with_changes(body=stmts))

  def _strip_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    if isinstance(node.body, SimpleStatementSuite):
      return node

    if not hasattr(node.body, "body"):
      return node

    new_body_stmts = []
    skip_next = False

    for i, stmt in enumerate(node.body.body):
      if skip_next:
        skip_next = False
        continue

      if self._is_super_init_stmt(stmt):
        # Check if the NEXT statement is an Escape Hatch Footer that should also be removed
        # (happens if the super call was wrapped due to warning/error)
        if i + 1 < len(node.body.body):
          next_stmt = node.body.body[i + 1]
          if self._is_escape_footer(next_stmt):
            skip_next = True
        continue

      new_body_stmts.append(stmt)

    return node.with_changes(body=node.body.with_changes(body=new_body_stmts))

  def _is_escape_footer(self, stmt: cst.CSTNode) -> bool:
    """Checks if a statement is the closure of an Escape Hatch."""
    # Ends usually look like: ... (Ellipsis) with a comment leading lines
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) > 0:
      if isinstance(stmt.body[0], cst.Expr) and isinstance(stmt.body[0].value, cst.Ellipsis):
        # Check comments for END_MARKER
        for line in stmt.leading_lines:
          if EscapeHatch.END_MARKER in line.comment.value:
            return True
    return False

  def _has_super_init(self, node: cst.FunctionDef) -> bool:
    if hasattr(node.body, "body"):
      for stmt in node.body.body:
        if self._is_super_init_stmt(stmt):
          return True
    return False

  def _is_super_init_stmt(self, stmt: cst.CSTNode) -> bool:
    """
    Detects if statement is `super().__init__()` or `super(Type, self).__init__()`.
    """
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
      small = stmt.body[0]
      if isinstance(small, cst.Expr) and isinstance(small.value, cst.Call):
        call = small.value
        # Check for .__init__()
        if isinstance(call.func, cst.Attribute) and call.func.attr.value == "__init__":
          receiver = call.func.value
          # Check for super(...)
          if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
            if receiver.func.value == "super":
              return True
    return False

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: List[str]) -> cst.FunctionDef:
    new_stmts = []
    for code in stmts_code:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:
        pass

    if isinstance(node.body, SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    existing = list(node.body.body)
    idx = (
      1
      if (
        existing
        and isinstance(existing[0], cst.SimpleStatementLine)
        and isinstance(existing[0].body[0], cst.Expr)
        and isinstance(existing[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString))
      )
      else 0
    )

    final_body = existing[:idx] + new_stmts + existing[idx:]
    return node.with_changes(body=node.body.with_changes(body=final_body))

  def _update_docstring(self, node: cst.FunctionDef, injected_args: List[Tuple[str, Optional[str]]]) -> cst.FunctionDef:
    body = node.body
    if not isinstance(body, IndentedBlock) or not body.body:
      return node

    stmt = body.body[0]
    if not isinstance(stmt, cst.SimpleStatementLine):
      return node
    if len(stmt.body) != 1 or not isinstance(stmt.body[0], cst.Expr):
      return node

    expr_node = stmt.body[0].value
    if not isinstance(expr_node, cst.SimpleString):
      return node

    raw_val = expr_node.value
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
      return node

    q_len = len(quote_style)
    if len(content_start) < 2 * q_len:
      return node

    inner_text = content_start[q_len:-q_len]
    new_text = self._modify_docstring_text(inner_text, injected_args)

    if new_text == inner_text:
      return node

    new_val = f"{prefix}{quote_style}{new_text}{quote_style}"
    new_expr = expr_node.with_changes(value=new_val)
    new_stmt = stmt.with_changes(body=[stmt.body[0].with_changes(value=new_expr)])
    new_body_stmts = list(body.body)
    new_body_stmts[0] = new_stmt

    return node.with_changes(body=body.with_changes(body=new_body_stmts))

  def _modify_docstring_text(self, text: str, args: List[Tuple[str, Optional[str]]]) -> str:
    lines = text.splitlines()
    if not lines:
      return text

    indent = ""
    for line in lines[1:]:
      if line.strip():
        indent = re.match(r"^\s*", line).group(0)
        break
    if not indent and len(lines) > 0:
      indent = "    "

    new_entries = []
    for name, _ in args:
      if re.search(rf"\b{name}\b\s*[:\(]", text) or re.search(rf"\b{name}\s+:", text):
        continue
      entry = f"{indent}{name}: Injected state argument."
      new_entries.append(entry)

    if not new_entries:
      return text

    args_match = re.search(r"(\n\s*Args:\s*\n)", text)
    if args_match:
      split_idx = args_match.end()
      block = "\n".join(new_entries)
      return text[:split_idx] + block + "\n" + text[split_idx:]

    param_match = re.search(r"(\n\s*Parameters\s*\n\s*[-=]+\s*\n)", text)
    if param_match:
      split_idx = param_match.end()
      block = "\n".join(new_entries)
      return text[:split_idx] + block + "\n" + text[split_idx:]

    if lines:
      final_indent = re.match(r"^\s*", lines[-1]).group(0) if lines[-1].strip() == "" else indent
      header = f"\n\n{final_indent}Args:\n"
      block = "\n".join([f"{final_indent}    {e.strip()}" for e in new_entries])
      return text + header + block + "\n"

    return text
