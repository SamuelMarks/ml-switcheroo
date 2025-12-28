""" 
Plugin Scaffolder. 

Generates valid Python source files for new hooks in the plugins directory. 
Used by the CLI to provide a starting point for complex logic implementations. 
Feature 083: Supports compiling declarative rules including complex operators into switch statements. 
Feature 084: Supports Preservative Updates (Body Extraction). 
Feature 085: Enforces PEP8 Filenaming (Snake Case). 
Feature 086: Auto-Wire Dictionary injection. 
"""

import re
import textwrap
import json
from pathlib import Path
from typing import List, Optional
import libcst as cst

from ml_switcheroo.core.dsl import PluginScaffoldDef, PluginType, Rule, LogicOp
from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal

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

# Adjusted templates to ensure exact whitespace control (no trailing spaces)
TEMPLATE_HEADER = '"""\n{doc}\n"""\nimport libcst as cst\nfrom ml_switcheroo.core.hooks import register_hook, HookContext\n\n'

TEMPLATE_FUNC_DEF_AUTO_WIRE = '@register_hook(trigger="{name}", auto_wire={auto_wire})\ndef {name}(node: {node_type}, ctx: HookContext) -> cst.CSTNode:\n    """\n    Plugin Hook: {doc}\n    """\n'

TEMPLATE_FUNC_DEF = '@register_hook("{name}")\ndef {name}(node: {node_type}, ctx: HookContext) -> cst.CSTNode:\n    """\n    Plugin Hook: {doc}\n    """\n'


class BodyExtractor(cst.CSTVisitor):
  """ 
  Extracts the body of a specific function definition. 
  Used to preserve user implementation logic during scaffolding updates. 
  """

  def __init__(self, func_name: str):
    self.func_name = func_name
    self.body_node: Optional[cst.BaseSuite] = None
    self.found = False

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """ 
    Visits function definitions to find the target hook. 
    If found, captures the body and stops recursion. 
    """
    if node.name.value == self.func_name:
      self.body_node = node.body
      self.found = True
      return False  # Stop visiting children
    return True


class PluginGenerator:
  """ 
  Writes Python plugin files to disk based on scaffold definitions. 
  """

  def __init__(self, plugins_dir: Path):
    """ 
    Initializes the generator. 

    Args: 
        plugins_dir: Target directory path. 
    """
    self.plugins_dir = plugins_dir

  def _to_snake_case(self, name: str) -> str:
    """ 
    Converts PascalCase or camelCase to snake_case for filenames. 
    e.g. MyPlugin -> my_plugin, HTTPResponse -> http_response 
    """
    # Handle simple lowercase existing
    if "_" in name and name.islower():
      return name

      # 1. Add underscore before capitals preceded by lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # 2. Add underscore before capitals preceded by uppercase/numbers 
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return s2

  def generate(self, scaffold: PluginScaffoldDef) -> bool:
    """ 
    Creates or updates a plugin file. 

    If the file exists, it attempts to preserve the existing function body logic
    while updating the wrapper (docstrings/decorators/imports). 

    Args: 
        scaffold: Definition model containing name, type, docs, and rules. 

    Returns: 
        bool: True if file was written/updated. 
    """
    clean_filename = self._to_snake_case(scaffold.name)
    filename = f"{clean_filename}.py"
    target_path = self.plugins_dir / filename
    preserved_body_code = None

    if target_path.exists():
      # Attempt Body Extraction
      try:
        old_code = target_path.read_text("utf-8")
        wrapper = cst.parse_module(old_code)
        extractor = BodyExtractor(scaffold.name)
        wrapper.visit(extractor)

        if extractor.found and extractor.body_node:
          preserved_body_code = self._render_body_without_docstring(extractor.body_node)
      except Exception as e:
        print(f"⚠️ Failed to parse existing plugin {filename}: {e}. Overwriting.")

    if not self.plugins_dir.exists():
      self.plugins_dir.mkdir(parents=True, exist_ok=True)

    content = self._build_content(scaffold, preserved_body_code)

    target_path.write_text(content, encoding="utf-8")
    return True

  def _render_body_without_docstring(self, body_node: cst.BaseSuite) -> str:
    """ 
    Serializes a Body Node into source code string, stripping the docstring. 

    Args: 
        body_node: The function body (IndentedBlock or SimpleStatementSuite). 

    Returns: 
        str: The indented source code of the body logic. 
    """
    stmts = []
    if isinstance(body_node, cst.IndentedBlock):
      stmts = list(body_node.body)
    elif isinstance(body_node, cst.SimpleStatementSuite):
      stmts = list(body_node.body)

      # Strip Docstring (First stmt is expression string)
    if stmts:
      first = stmts[0]
      idx = 0
      is_doc = False
      if isinstance(first, cst.SimpleStatementLine) and len(first.body) == 1:
        expr = first.body[0]
        if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
          is_doc = True

      if is_doc:
        # Skip the docstring statement
        stmts = stmts[1:]

    if not stmts:
      # Empty body or just docstring -> return 'return node' default or pass
      return "    return node"

      # Render back to string using a temporary module container
    # We construct a module from the statements to leverage LibCST's code generation
    temp_mod = cst.Module(body=stmts)
    code = temp_mod.code

    # The code extracted might contain inconsistent indentation
    # We strip common indentation and enforce standardized 4-space indent
    dedented = textwrap.dedent(code)
    indented = textwrap.indent(dedented, "    ")
    # Strip Trailing newlines to ensure clean insertion
    return indented.rstrip()

  def _build_content(self, scaffold: PluginScaffoldDef, preserved_body: Optional[str] = None) -> str:
    """Constructs the full python source for the file."""
    parts = []

    # 1. Header
    parts.append(TEMPLATE_HEADER.format(doc=scaffold.doc))

    # 2. Helpers (only if rules present or preserved code relied on them?) 
    # Simplest strategy: Always include helpers if rules are defined in scaffold. 
    if scaffold.rules:
      parts.append(HELPER_LOGIC)

      # 3. Function Definition
    node_type = "cst.Call" if scaffold.type == PluginType.CALL else "cst.CSTNode"

    if scaffold.auto_wire:
      # Use json dumps to get double quotes matching test expectations
      # and generally being standard for JSON-like config in Python.
      import json
      json_str = json.dumps(scaffold.auto_wire)
      # Fix Python literals
      safe_repr = json_str.replace("true", "True").replace("false", "False").replace("null", "None")

      parts.append(TEMPLATE_FUNC_DEF_AUTO_WIRE.format(
        name=scaffold.name,
        doc=scaffold.doc,
        node_type=node_type,
        auto_wire=safe_repr
      ))
    else:
      parts.append(TEMPLATE_FUNC_DEF.format(name=scaffold.name, doc=scaffold.doc, node_type=node_type))

      # 4. Body (Preserved or Generated)
    if preserved_body and preserved_body.strip():
      # Ensure newline before body
      if not preserved_body.startswith("\n"):
        parts.append("\n")
      parts.append(preserved_body)
      # Ensure newline at end
      if not preserved_body.endswith("\n"):
        parts.append("\n")
    else:
      body = self._generate_body_logic(scaffold.rules)
      parts.append(body)

    return "".join(parts)

  def _generate_body_logic(self, rules: List[Rule]) -> str:
    """Compiles declarative rules into Python if statements."""
    if not rules:
      return "    # TODO: Implement custom logic\n    return node\n"

    lines = []
    lines.append("    # Auto-Generated Conditional Logic")

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

    for i, rule in enumerate(rules):
      keyword = "if" if i == 0 else "elif"
      val_repr = repr(rule.is_val)
      py_op = op_map.get(rule.op, "==")

      lines.append(f'    val_{i} = _get_kwarg_value(node, "{rule.if_arg}")')

      # Safety Wrapper for None comparisons
      if rule.op in [LogicOp.GT, LogicOp.LT, LogicOp.GTE, LogicOp.LTE]:
        lines.append(f"    {keyword} val_{i} is not None and val_{i} {py_op} {val_repr}:")
      else:
        lines.append(f"    {keyword} val_{i} {py_op} {val_repr}:")

      lines.append(f'        new_func = _create_dotted_name("{rule.use_api}")')
      lines.append(f"        return node.with_changes(func=new_func)")

    lines.append("    ")
    lines.append("    return node\n")

    return "\n".join(lines)
