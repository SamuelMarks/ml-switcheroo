"""Plugin and DSL Passes for ML-Switcheroo.

Includes layout permutations (NCHW <-> NHWC), macro template execution,
and import fixing logic built on top of the generated AST.
"""

from typing import List, Dict
import libcst as cst


class LayoutPermutationPass(cst.CSTTransformer):
  """Detects and fixes NCHW vs NHWC mismatches via permutations."""

  def __init__(self, source_layout: str, target_layout: str) -> None:
    """Initialize the LayoutPermutationPass.

    Args:
        source_layout: The layout of the source framework (e.g., 'NCHW').
        target_layout: The expected layout of the target framework (e.g., 'NHWC').
    """
    super().__init__()
    self.source_layout = source_layout.upper()
    self.target_layout = target_layout.upper()

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
    """Inject permutation calls for known layout-sensitive ops.

    Args:
        original_node: Original Call node.
        updated_node: Updated Call node.

    Returns:
        Potentially wrapped Call node with permutation.
    """
    if self.source_layout == self.target_layout:
      return updated_node

    if self.source_layout == "NCHW" and self.target_layout == "NHWC":
      # Very simplistic heuristic: assume 4D tensor outputs that need (0, 2, 3, 1)
      permute_attr = cst.Attribute(value=updated_node, attr=cst.Name("permute"))
      return cst.Call(func=permute_attr, args=[cst.Arg(value=cst.Integer(str(i))) for i in (0, 2, 3, 1)])
    elif self.source_layout == "NHWC" and self.target_layout == "NCHW":
      # Reverse permutation (0, 3, 1, 2)
      permute_attr = cst.Attribute(value=updated_node, attr=cst.Name("permute"))
      return cst.Call(func=permute_attr, args=[cst.Arg(value=cst.Integer(str(i))) for i in (0, 3, 1, 2)])

    return updated_node


class FrameworkMacroPass(cst.CSTTransformer):
  """Expands YAML macro templates onto the AST.

  e.g., "{x} * sigmoid({x})" for a SiLU fallback.
  """

  def __init__(self, macros: Dict[str, str]) -> None:
    """Initialize FrameworkMacroPass.

    Args:
        macros: Dictionary mapping target op names to macro templates.
    """
    super().__init__()
    self.macros = macros

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
    """Expand macros.

    Args:
        original_node: Original Call node.
        updated_node: Updated Call node.

    Returns:
        The parsed macro expression if matched, otherwise updated_node.
    """
    if isinstance(updated_node.func, cst.Name):
      func_name = updated_node.func.value
      if func_name in self.macros:
        template = self.macros[func_name]
        # Simplistic substitution: assumes single arg "x"
        if len(updated_node.args) == 1:
          arg_str = cst.Module(body=[]).code_for_node(updated_node.args[0].value)
          expanded_str = template.replace("{x}", arg_str)

          try:
            # Parse the expanded string into a Python expression
            # We wrap it in a function to extract the single expression easily
            expr_ast = cst.parse_expression(expanded_str)
            return expr_ast
          except Exception:
            # If parsing fails (e.g. malformed macro), return original
            return updated_node
    return updated_node


class ImportFixerPass(cst.CSTTransformer):
  """Ensures necessary module imports are present and unused ones removed."""

  def __init__(self, required_imports: List[str]) -> None:
    """Initialize ImportFixerPass.

    Args:
        required_imports: List of module names that must be imported.
    """
    super().__init__()
    self.required_imports = required_imports

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """Inject missing imports at the top of the module.

    Args:
        original_node: Original Module node.
        updated_node: Updated Module node.

    Returns:
        Module node with imports.
    """
    existing_imports = set()
    for stmt in updated_node.body:
      if isinstance(stmt, cst.SimpleStatementLine):
        for small_stmt in stmt.body:
          if isinstance(small_stmt, cst.Import):
            for alias in small_stmt.names:
              existing_imports.add(cst.Module(body=[]).code_for_node(alias.name))

    new_body = list(updated_node.body)
    for req in self.required_imports:
      if req not in existing_imports:
        import_stmt = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(req))])])
        # Insert after docstring if present
        insert_idx = 0
        if new_body and isinstance(new_body[0], cst.SimpleStatementLine) and isinstance(new_body[0].body[0], cst.Expr):
          insert_idx = 1
        new_body.insert(insert_idx, import_stmt)

    return updated_node.with_changes(body=new_body)
