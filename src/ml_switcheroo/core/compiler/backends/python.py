"""Python code synthesis backend for compiling LogicalGraph IR into Python CST modules."""

import typing

import libcst as cst

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.python_strategies.forward import build_forward
from ml_switcheroo.core.compiler.backends.python_strategies.imports import generate_imports
from ml_switcheroo.core.compiler.backends.python_strategies.init import build_init
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, topological_sort
from ml_switcheroo.frameworks.base import get_adapter

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class ClassBodyReplacer(cst.CSTTransformer):
  """Transformer to swap __init__ and forward methods in a target class."""

  def __init__(
    self,
    target_class: str,
    new_init: cst.FunctionDef,
    new_forward: cst.FunctionDef,
  ) -> None:
    """Initialize ClassBodyReplacer.

    Args:
        target_class: The name of the class to replace.
        new_init: The AST node for the new __init__ method.
        new_forward: The AST node for the new forward method.
    """
    self.target_class = target_class
    self.new_init = new_init
    self.new_forward = new_forward
    self.found = False

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    """Process and modify a class definition.

    Args:
        original_node: The original CST ClassDef node.
        updated_node: The updated CST ClassDef node.

    Returns:
        The modified CST ClassDef node.
    """
    if original_node.name.value == self.target_class:
      self.found = True

      # Handle one-line classes "class A: pass" which LibCST parses as SimpleStatementSuite
      current_body = updated_node.body
      stmts_list: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []

      if isinstance(current_body, cst.SimpleStatementSuite):
        # Convert inline body to block statements
        # Inline bodies contain SmallStatements (e.g. Pass, Expr).
        # We wrap them in SimpleStatementLine logic.
        for small_stmt in current_body.body:
          stmts_list.append(cst.SimpleStatementLine(body=[small_stmt]))
      else:
        stmts_list = [s for s in current_body.body if isinstance(s, (cst.SimpleStatementLine, cst.BaseCompoundStatement))]

      new_body_stmts: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
      replacements = {
        "__init__": self.new_init,
        "forward": self.new_forward,
        "call": self.new_forward,
        "__call__": self.new_forward,
      }
      injected = set()

      # Filter out old methods and replace
      for stmt in stmts_list:
        if isinstance(stmt, cst.FunctionDef):
          fname = stmt.name.value
          if fname in replacements:
            # Only inject replacement once per function key
            if fname not in injected:
              new_body_stmts.append(replacements[fname])
              injected.add(fname)
            # Skip original dict items
          else:
            new_body_stmts.append(stmt)
        else:
          new_body_stmts.append(stmt)

      # Inject missing required methods
      if "__init__" not in injected:
        new_body_stmts.insert(0, self.new_init)

      inference_hit = any(x in injected for x in ["forward", "call", "__call__"])
      if not inference_hit:
        new_body_stmts.append(self.new_forward)

      # Ensure we return an IndentedBlock even if source was SimpleStatementSuite
      new_block = cst.IndentedBlock(body=new_body_stmts)
      return updated_node.with_changes(body=new_block)

    return updated_node


class PythonBackend(CompilerBackend):
  """Synthesize a Python CST Module from a LogicalGraph."""

  def __init__(self, framework: str = "torch", semantics: "SemanticsManager | None" = None) -> None:
    """Initialize PythonBackend with target framework and semantics manager.

    Args:
        framework: Target framework identifier (default: 'torch').
        semantics: Optional SemanticsManager instance for resolving API variants.
    """
    self.framework = framework
    self.semantics = semantics
    adapter = get_adapter(framework)
    if adapter:
      self.traits = adapter.structural_traits
    else:
      from ml_switcheroo.semantics.schema import StructuralTraits

      self.traits = StructuralTraits()

  def _is_stateful_layer(self, node: LogicalNode) -> bool:
    """Check whether a LogicalNode represents a stateful neural layer.

    Args:
        node: The LogicalNode to inspect.

    Returns:
        True if the node is a stateful layer, False otherwise.
    """
    from ml_switcheroo.core.compiler.backends.python_strategies.init import is_stateful_layer

    return is_stateful_layer(node)

  def _generate_layer_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
    """Generate the LibCST layer initialization statement for a LogicalNode.

    Args:
        node: The LogicalNode representing the layer to initialize.

    Returns:
        Generated LibCST SimpleStatementLine for initializing the layer.
    """
    from ml_switcheroo.core.compiler.backends.python_strategies.init import generate_layer_init

    return generate_layer_init(node, self.framework, self.semantics)

  def compile(self, graph: LogicalGraph) -> str:
    """Compile the given LogicalGraph to a Python string.

    Args:
        graph: The source LogicalGraph.

    Returns:
        Python source code string.
    """
    # Use graph name if available, else default
    name = graph.name if graph.name else "SwitcherooNet"
    return self.generate(graph, class_name=name)

  def generate(
    self,
    graph: LogicalGraph,
    class_name: str = "SwitcherooNet",
    original_tree: cst.Module | None = None,
  ) -> str:
    """Generate code for the given graph and class name.

    Args:
        graph: The LogicalGraph to compile.
        class_name: The name for the generated Python class.
        original_tree: An optional parsed CST tree to patch.

    Returns:
        The generated Python source code string.
    """
    ordered_nodes = topological_sort(graph)
    # Ensure graph input/output conventions are respected
    init_func = build_init(ordered_nodes, self.framework, self.semantics, self.traits)
    forward_func = build_forward(ordered_nodes, self.framework, self.semantics, self.traits, self)

    if original_tree:
      replacer = ClassBodyReplacer(class_name, init_func, forward_func)
      new_tree = original_tree.visit(replacer)
      if replacer.found:
        return new_tree.code

    body: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
    body.extend(generate_imports(self.framework))

    base_class = (
      getattr(self.traits, "module_base", None) or getattr(self.traits, "fallback_module_base", None) or "nn.Module"
    )
    short_pref_rule = getattr(self.traits, "module_base_short_prefix", None)
    if short_pref_rule and base_class.startswith(short_pref_rule[0]):
      base_class = short_pref_rule[1] + base_class[len(short_pref_rule[0]) :]

    override_rule = getattr(self.traits, "model_base_override", None)
    if override_rule and base_class == override_rule[0]:
      base_class = override_rule[1]

    # To add spacing without appending invalid EmptyLine tokens to the body list,
    # we attach leading_lines to the class definition.
    class_def = cst.ClassDef(
      name=cst.Name(class_name),
      bases=[cst.Arg(value=cst.parse_expression(base_class))],
      leading_lines=[cst.EmptyLine(newline=cst.Newline())],
      body=cst.IndentedBlock(
        body=[
          init_func,
          # Attach empty line to forward_func for separation inside class
          forward_func.with_changes(leading_lines=[cst.EmptyLine(newline=cst.Newline())]),
        ]
      ),
    )

    body.append(class_def)

    # LibCST module expects sequence of statements
    module = cst.Module(body=body)
    return module.code
