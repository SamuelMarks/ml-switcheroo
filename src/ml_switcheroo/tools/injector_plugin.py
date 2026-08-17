"""Plugin Scaffolder.

Generates valid Python source files for new hooks in the plugins directory.
Used by the CLI to provide a starting point for complex logic implementations.
Feature 083: Supports compiling declarative rules including complex operators into switch statements.
Feature 084: Supports Preservative Updates (Body Extraction).
Feature 085: Enforces PEP8 Filenaming (Snake Case).
Feature 086: Auto-Wire Dictionary injection.
"""

from pathlib import Path
from typing import List, Optional
import libcst as cst

from ml_switcheroo.core.dsl import PluginScaffoldDef, PluginType, Rule, LogicOp

# Helper logic injected into plugins generated with rules
HELPER_LOGIC = """
def _get_kwarg_value(node: cst.Call, arg_name: str):
    for arg in node.args:
        if arg.keyword and arg.keyword.value == arg_name:
             return _node_to_literal(arg.value)
    return None

def _node_to_literal(node):
    if isinstance(node, cst.Integer): return int(node.value)
    if isinstance(node, cst.Float): return float(node.value)
    if isinstance(node, cst.SimpleString): return node.value.strip("'").strip('"')
    if isinstance(node, cst.Name):
         if node.value == "True": return True
         if node.value == "False": return False
         if node.value == "None": return None
    return None

def _create_dotted_name(name_str: str) -> cst.BaseExpression:
    parts = name_str.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
        node = cst.Attribute(value=node, attr=cst.Name(part))
    return node
"""


class NameMangler:
  """Utility to safely transform string cases without regex."""

  @staticmethod
  def to_snake_case(name: str) -> str:
    """Converts PascalCase or camelCase to snake_case.

    Args:
        name (str): The input name.

    Returns:
        str: The snake_case version of the name.
    """
    if "_" in name and name.islower():
      return name

    result: List[str] = []
    for i, char in enumerate(name):
      if char.isupper():
        if i > 0:
          prev_char = name[i - 1]
          next_char = name[i + 1] if i + 1 < len(name) else ""
          if (
            prev_char.islower() or prev_char.isdigit() or (next_char.islower() and next_char.isalpha())
          ):  # pragma: no branch
            result.append("_")
      result.append(char.lower())

    return "".join(result)


class BodyExtractor(cst.CSTVisitor):
  """Extracts the body of a specific function definition.

  Used to preserve user implementation logic during scaffolding updates.
  """

  def __init__(self, func_name: str):
    """Initializes the extractor to look for a function by name.

    Args:
        func_name (str): The name of the function to extract.

    """
    self.func_name = func_name
    self.body_node: Optional[cst.BaseSuite] = None
    self.found = False

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Visits function definitions to find the target hook.

    If found, captures the body and stops recursion.

    Args:
        node (cst.FunctionDef): The function definition node.

    Returns:
        Optional[bool]: False if found to stop recursion, True otherwise.

    """
    if node.name.value == self.func_name:
      self.body_node = node.body
      self.found = True
      return False  # Stop visiting children
    return True


class PluginGenerator:
  """Writes Python plugin files to disk based on scaffold definitions."""

  def __init__(self, plugins_dir: Path):
    """Initializes the generator.

    Args:
        plugins_dir: Target directory path.

    """
    self.plugins_dir = plugins_dir

  def generate(self, scaffold: PluginScaffoldDef) -> bool:
    """Creates or updates a plugin file.

    If the file exists, it attempts to preserve the existing function body logic
    while updating the wrapper (docstrings/decorators/imports).

    Args:
        scaffold: Definition model containing name, type, docs, and rules.

    Returns:
        bool: True if file was written/updated.

    """
    clean_filename = NameMangler.to_snake_case(scaffold.name)
    filename = f"{clean_filename}.py"
    target_path = self.plugins_dir / filename
    preserved_body_node = None

    if target_path.exists():
      try:
        old_code = target_path.read_text("utf-8")
        wrapper = cst.parse_module(old_code)
        extractor = BodyExtractor(scaffold.name)
        wrapper.visit(extractor)

        if extractor.found and extractor.body_node:  # pragma: no branch
          preserved_body_node = extractor.body_node
      except Exception as e:
        print(f"⚠️ Failed to parse existing plugin {filename}: {e}. Overwriting.")

    if not self.plugins_dir.exists():
      self.plugins_dir.mkdir(parents=True, exist_ok=True)

    content = self._build_cst_content(scaffold, preserved_body_node)

    target_path.write_text(content, encoding="utf-8")
    return True

  def _build_cst_content(self, scaffold: PluginScaffoldDef, preserved_body: Optional[cst.BaseSuite] = None) -> str:
    """Constructs the full python source for the file using CST.

    Args:
        scaffold: The plugin definition.
        preserved_body: Optional preserved source code for the body.

    Returns:
        str: The complete file source string.

    """
    # Build initial module layout
    base_module_str = (
      f'"""\n{scaffold.doc}\n"""\nimport libcst as cst\nfrom ml_switcheroo.core.hooks import register_hook, HookContext\n'
    )
    module = cst.parse_module(base_module_str)

    # 2. Helpers
    helpers_stmts: List[cst.BaseStatement] = []
    if scaffold.rules:
      helpers_mod = cst.parse_module(HELPER_LOGIC)
      helpers_stmts = list(helpers_mod.body)

    # 3. Function Definition
    node_type = "cst.Call" if scaffold.type == PluginType.CALL else "cst.CSTNode"

    if scaffold.auto_wire:
      import json

      json_str = json.dumps(scaffold.auto_wire)
      safe_repr = json_str.replace("true", "True").replace("false", "False").replace("null", "None")
      func_stub = f'@register_hook(trigger="{scaffold.name}", auto_wire={safe_repr})\ndef {scaffold.name}(node: {node_type}, ctx: HookContext) -> cst.CSTNode:\n    pass'
    else:
      func_stub = f'@register_hook("{scaffold.name}")\ndef {scaffold.name}(node: {node_type}, ctx: HookContext) -> cst.CSTNode:\n    pass'

    func_def = cst.parse_statement(func_stub)

    # 4. Body
    doc_stmt = cst.parse_statement(f'"""\n    Plugin Hook: {scaffold.doc}\n    """')

    if preserved_body:
      stmts = []
      if isinstance(preserved_body, cst.IndentedBlock):
        stmts = list(preserved_body.body)
      elif isinstance(preserved_body, cst.SimpleStatementSuite):  # pragma: no branch
        stmts = [cst.SimpleStatementLine(body=list(preserved_body.body))]

      # Strip existing docstring
      if stmts:  # pragma: no branch
        first = stmts[0]
        is_doc = False
        if isinstance(first, cst.SimpleStatementLine) and len(first.body) == 1:
          expr = first.body[0]
          if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
            is_doc = True
        if is_doc:
          stmts = stmts[1:]

      if not stmts:
        stmts = [cst.parse_statement("return node")]

      new_body = [doc_stmt] + stmts
      func_def = func_def.with_changes(body=cst.IndentedBlock(body=new_body))
    else:
      generated_stmts = self._generate_cst_body_logic(scaffold.rules)
      func_def = func_def.with_changes(body=cst.IndentedBlock(body=[doc_stmt] + generated_stmts))

    final_body = list(module.body) + helpers_stmts + [func_def]
    module = module.with_changes(body=final_body)
    return module.code

  def _generate_cst_body_logic(self, rules: List[Rule]) -> List[cst.BaseStatement]:
    """Compiles declarative rules into CST statements.

    Args:
        rules: List of dispatch rules.

    Returns:
        List[cst.BaseStatement]: Generated python CST statements.
    """
    if not rules:
      mod = cst.parse_module("def __temp():\n    # TODO: Implement custom logic\n    return node\n")
      func_def = mod.body[0]
      if isinstance(func_def, cst.FunctionDef) and isinstance(func_def.body, cst.IndentedBlock):
        return list(func_def.body.body)
      return []

    op_map = {
      LogicOp.EQ: "==",
      LogicOp.NEQ: "!=",
      LogicOp.GT: ">",
      LogicOp.LT: "<",
      LogicOp.GTE: ">=",
      LogicOp.LTE: "<=",
      LogicOp.IN: "in",
      LogicOp.NOT_IN: "not in",
    }

    lines = []
    lines.append("def __temp():")
    lines.append("    # Auto-Generated Conditional Logic")

    for i, rule in enumerate(rules):
      keyword = "if"
      val_repr = repr(rule.is_val)
      py_op = op_map.get(rule.op, "==")

      lines.append(f'    val_{i} = _get_kwarg_value(node, "{rule.if_arg}")')

      if rule.op in [LogicOp.GT, LogicOp.LT, LogicOp.GTE, LogicOp.LTE]:
        lines.append(f"    {keyword} val_{i} is not None and val_{i} {py_op} {val_repr}:")
      else:
        lines.append(f"    {keyword} val_{i} {py_op} {val_repr}:")

      lines.append(f'        new_func = _create_dotted_name("{rule.use_api}")')
      lines.append("        return node.with_changes(func=new_func)")

    lines.append("    return node")

    code = "\n".join(lines)
    mod = cst.parse_module(code)
    func_def = mod.body[0]
    if isinstance(func_def, cst.FunctionDef) and isinstance(func_def.body, cst.IndentedBlock):
      return list(func_def.body.body)
    return []
