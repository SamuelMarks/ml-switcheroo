"""
AST Visualization Utility.

This module provides the `MermaidGenerator`, a LibCST visitor that traverses
an Abstract Syntax Tree and converts it into a Mermaid.js Graph diagram.
It is used to generate "Before" and "After" snapshots of the code structure
during transpilation for debugging and documentation.

The generator applies specific branding colors and handles complex node
representation (like dotted imports) robustly.
"""

import uuid
from typing import List, Optional

import libcst as cst


class MermaidGenerator(cst.CSTVisitor):
  """
  Generates a Mermaid Graph TD string from a CST Node tree.

  It traverses the tree and emits nodes and edges formatted with specific
  branding colors.
  """

  # Branding Colors
  COLORS = {
    "blue": "#4285f4",
    "green": "#34a853",
    "yellow": "#f9ab00",
    "red": "#ea4335",
    "navy": "#20344b",
    "white": "#ffffff",
    "halftone_blue": "#57caff",
    "halftone_green": "#5cdb6d",
    "halftone_yellow": "#ffd427",
    "halftone_red": "#ff7daf",
  }

  # Mermaid Style Definitions
  # Note: Class names mapped to avoid Mermaid keywords (call, import, etc.)
  STYLES = f"""
    %% Styles
    classDef default font-family:'Google Sans Normal',color:{COLORS["navy"]},stroke:{COLORS["navy"]};
    classDef modNode fill:{COLORS["navy"]},stroke:{COLORS["navy"]},color:{COLORS["white"]},rx:5px,font-family:'Google Sans Medium';
    classDef classNode fill:{COLORS["red"]},stroke:{COLORS["navy"]},color:{COLORS["white"]},rx:5px,font-family:'Google Sans Medium';
    classDef funcNode fill:{COLORS["blue"]},stroke:{COLORS["navy"]},color:{COLORS["white"]},rx:5px,font-family:'Google Sans Medium';
    classDef callNode fill:{COLORS["green"]},stroke:{COLORS["navy"]},stroke-width:2px,color:{COLORS["white"]},rx:5px,font-family:'Roboto Mono Normal';
    classDef stmtNode fill:{COLORS["white"]},stroke:{COLORS["navy"]},stroke-dasharray: 2 2,color:{COLORS["navy"]},font-family:'Roboto Mono Normal';
    classDef argNode fill:{COLORS["yellow"]},stroke:{COLORS["navy"]},color:{COLORS["navy"]},rx:2px,font-size:10px;
    classDef impNode fill:{COLORS["halftone_blue"]},stroke:{COLORS["navy"]},color:{COLORS["navy"]},rx:5px;
    """

  def __init__(self):
    """Initializes the generator with empty buffers."""
    self.nodes: List[str] = []
    self.edges: List[str] = []
    self.stack: List[str] = []
    # Dummy module used for code generation logic
    self._renderer = cst.Module([])

  def generate(self, tree: cst.CSTNode) -> str:
    """
    Converts a CST Node into a Mermaid graph definition string.

    Args:
        tree (cst.CSTNode): The root node of the tree to visualize.

    Returns:
        str: A complete Mermaid.js graph definition including styles.
    """
    self.nodes = []
    self.edges = []
    self.stack = []

    # Start traversal
    tree.visit(self)

    return f"graph TD\n{self.STYLES}\n" + "\n".join(self.nodes + self.edges)

  def _add_node(self, label: str, style_class: str = "default") -> str:
    """
    Helper to register a node in the graph and link it to its parent.

    Args:
        label (str): Text display for the node.
        style_class (str): CSS class for styling (defined in STYLES).

    Returns:
        str: The unique ID generated for this node.
    """
    node_id = f"n{uuid.uuid4().hex[:8]}"
    # Clean label for mermaid compliance (escape quotes)
    clean_label = label.replace('"', "'").replace("\n", " ").strip()

    # Truncate overly long labels
    if len(clean_label) > 40:
      clean_label = clean_label[:37] + "..."

    self.nodes.append(f'{node_id}["{clean_label}"]:::{style_class}')

    # Link to parent if traversing recursively
    if self.stack:
      parent = self.stack[-1]
      self.edges.append(f"{parent} --> {node_id}")

    return node_id

  def _node_to_str(self, node: cst.CSTNode) -> str:
    """
    Robustly extracts a string representation of a Name, Attribute, or complex expression.
    Avoids LibCST code generation for simple cases to prevent crashes on detached nodes.

    Args:
        node (cst.CSTNode): The node to stringify.

    Returns:
        str: The code string.
    """
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      return f"{self._node_to_str(node.value)}.{node.attr.value}"
    elif isinstance(node, cst.Integer):
      return node.value
    elif isinstance(node, cst.Float):
      return node.value
    elif isinstance(node, cst.SimpleString):
      return node.value

    # Fallback to code generator for complex expressions (e.g. calls, tuples)
    try:
      return self._renderer.code_for_node(node).strip()
    except Exception:
      return f"<{type(node).__name__}>"

  def visit_Module(self, node: cst.Module) -> Optional[bool]:
    """Visits Module root."""
    id = self._add_node("Module", "modNode")
    self.stack.append(id)
    return True

  def leave_Module(self, node: cst.Module) -> None:
    """Leaves Module root."""
    if self.stack:
      self.stack.pop()

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """Visits Class Definitions."""
    id = self._add_node(f"Class: {node.name.value}", "classNode")
    self.stack.append(id)
    return True

  def leave_ClassDef(self, node: cst.ClassDef) -> None:
    """Leaves Class Definitions."""
    if self.stack:
      self.stack.pop()

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Visits Function Definitions."""
    id = self._add_node(f"Def: {node.name.value}", "funcNode")
    self.stack.append(id)
    return True

  def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
    """Leaves Function Definitions."""
    if self.stack:
      self.stack.pop()

  def visit_Call(self, node: cst.Call) -> Optional[bool]:
    """Visits Function Calls."""
    # Helper to get name
    try:
      name = self._node_to_str(node.func)
      if "(" in name:
        # Fallback if renderer made something too complex
        name = "Call"
      else:
        name = f"{name}()"
    except Exception:
      name = "Call"

    id = self._add_node(name, "callNode")
    self.stack.append(id)
    return True  # Visit args

  def leave_Call(self, node: cst.Call) -> None:
    """Leaves Function Calls."""
    if self.stack:
      self.stack.pop()

  def visit_Arg(self, node: cst.Arg) -> Optional[bool]:
    """Visits Arguments inside a Call."""
    # Try to summarize arg key
    label = "arg"
    if node.keyword:
      label = f"{node.keyword.value}="

    # Check for simple literals to inline them into the Arg label
    is_simple = False
    if isinstance(node.value, (cst.Name, cst.Integer, cst.Float, cst.SimpleString)):
      try:
        val_code = self._node_to_str(node.value)
        label += val_code
        is_simple = True
      except Exception:
        pass

    # Always add node and push to stack to maintain symmetry for leave_Arg
    id = self._add_node(label, "argNode")
    self.stack.append(id)

    if is_simple:
      return False  # Don't recurse into simple value nodes

    return True

  def leave_Arg(self, node: cst.Arg) -> None:
    """Leaves Arguments."""
    if self.stack:
      self.stack.pop()

  def visit_Import(self, node: cst.Import) -> Optional[bool]:
    """Visits Import statements (collapsing them into single nodes)."""
    names = [n.name.value for n in node.names if isinstance(n.name, cst.Name)]
    if not names:
      names = ["..."]
    self._add_node(f"Import {', '.join(names)}", "impNode")
    return False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
    """Visits From-Import statements (collapsed)."""
    mod = self._node_to_str(node.module) if node.module else "."
    names = []
    if isinstance(node.names, cst.ImportStar):
      names.append("*")
    else:
      for n in node.names:
        if hasattr(n, "name") and hasattr(n.name, "value"):
          names.append(n.name.value)

    display_names = ", ".join(names[:3])
    if len(names) > 3:
      display_names += "..."

    self._add_node(f"From {mod} Import {display_names}", "impNode")
    return False

  def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
    """Visits Assignment statements."""
    id = self._add_node("Assign (=)", "stmtNode")
    self.stack.append(id)
    return True

  def leave_Assign(self, node: cst.Assign) -> None:
    """Leaves Assignment statements."""
    if self.stack:
      self.stack.pop()
