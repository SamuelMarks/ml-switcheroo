"""
Docstring Rewriting Logic for Function Definitions.

This module provides the `FuncDocstringMixin` used by the `structure_func` rewriter.
It handles parsing, parsing, and updating Python docstrings (Google/NumPy style)
to document arguments injected during transpilation (e.g., `rngs`).
"""

import re
from typing import List, Tuple, Optional
import libcst as cst
from libcst import IndentedBlock, SimpleStatementLine


class FuncDocstringMixin:
  """
  Mixin for updating function docstrings.

  Provides methods to inject parameter descriptions into existing docstrings,
  creating `Args:` or `Parameters` sections if they don't exist.
  """

  def _update_docstring(self, node: cst.FunctionDef, injected_args: List[Tuple[str, Optional[str]]]) -> cst.FunctionDef:
    """
    Updates the function docstring to describe injected arguments.

    Supports Google and NumPy style docstrings. Creates an ``Args:`` section
    if one does not exist.

    Args:
        node: The function definition.
        injected_args: List of (name, type) tuples that were added to signature.

    Returns:
        The function definition with updated docstring.
    """
    body = node.body
    if not isinstance(body, IndentedBlock) or not body.body:
      return node

    stmt = body.body[0]
    if not isinstance(stmt, SimpleStatementLine):
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
    """
    Applies regex replacements to inject argument descriptions into docstring text.

    Args:
        text: The inner text of the docstring.
        args: Arguments to document.

    Returns:
        The modified text.
    """
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
