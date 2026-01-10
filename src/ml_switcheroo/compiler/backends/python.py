"""
Python Source Code Backend.

This module implements a Compiler Backend that synthesizes Python source code
from the Logical Graph Internal Representation via LibCST.
"""

from typing import Any, Dict, List, Optional, Union
import libcst as cst
from libcst import matchers as m

from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, topological_sort


class ClassBodyReplacer(cst.CSTTransformer):
  """
  Transformer to swap __init__ and forward methods in a target class.
  """

  def __init__(
    self,
    target_class: str,
    new_init: cst.FunctionDef,
    new_forward: cst.FunctionDef,
  ) -> None:
    self.target_class = target_class
    self.new_init = new_init
    self.new_forward = new_forward
    self.found = False

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    if original_node.name.value == self.target_class:
      self.found = True

      # Handle one-line classes "class A: pass" which LibCST parses as SimpleStatementSuite
      current_body = updated_node.body
      stmts_list: List[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]] = []

      if isinstance(current_body, cst.SimpleStatementSuite):
        # Convert inline body to block statements
        # Inline bodies contain SmallStatements (e.g. Pass, Expr).
        # We wrap them in SimpleStatementLine logic.
        for stmt in current_body.body:
          if isinstance(stmt, (cst.Pass, cst.Expr, cst.Assign, cst.AnnAssign, cst.Return)):
            # SmallStatement -> SimpleStatementLine
            stmts_list.append(cst.SimpleStatementLine(body=[stmt]))
      elif isinstance(current_body, cst.IndentedBlock):
        stmts_list = list(current_body.body)

      new_body_stmts = []
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
  """
  Synthesizes a Python CST Module from a LogicalGraph.
  """

  def __init__(self, framework: str = "torch", semantics: Any = None) -> None:
    self.framework = framework

  def compile(self, graph: LogicalGraph) -> str:
    # Use graph name if available, else default
    name = graph.name if graph.name else "SwitcherooNet"
    return self.generate(graph, class_name=name)

  def generate(
    self,
    graph: LogicalGraph,
    class_name: str = "SwitcherooNet",
    original_tree: Optional[cst.Module] = None,
  ) -> str:
    ordered_nodes = topological_sort(graph)
    # Ensure graph input/output conventions are respected
    init_func = self._build_init(ordered_nodes)
    forward_func = self._build_forward(ordered_nodes)

    if original_tree:
      replacer = ClassBodyReplacer(class_name, init_func, forward_func)
      new_tree = original_tree.visit(replacer)
      if isinstance(new_tree, cst.Module) and replacer.found:
        return new_tree.code

    body: List[cst.CSTNode] = []
    body.extend(self._generate_imports())

    base_class = "nn.Module"
    if self.framework in ["jax", "flax", "flax_nnx"]:
      base_class = "nnx.Module"
    elif self.framework == "keras":
      base_class = "keras.Model"

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

  def _generate_imports(self) -> List[cst.SimpleStatementLine]:
    if self.framework == "torch":
      return [
        cst.parse_statement("import torch"),
        cst.parse_statement("import torch.nn as nn"),
      ]
    elif self.framework in ["jax", "flax", "flax_nnx"]:
      return [
        cst.parse_statement("from flax import nnx"),
        cst.parse_statement("import jax.numpy as jnp"),
      ]
    return []

  def _build_init(self, nodes: List[LogicalNode]) -> cst.FunctionDef:
    stmts: List[cst.BaseStatement] = []
    if self.framework in ["torch", "keras"]:
      stmts.append(cst.parse_statement("super().__init__()"))

    for node in nodes:
      if self._is_stateful_layer(node):
        assignment = self._generate_layer_init(node)
        stmts.append(assignment)

    params = [cst.Param(name=cst.Name("self"))]
    if self.framework in ["jax", "flax", "flax_nnx"]:
      params.append(
        cst.Param(
          name=cst.Name("rngs"),
          annotation=cst.Annotation(cst.parse_expression("nnx.Rngs")),
        )
      )

    if not stmts:
      stmts.append(cst.parse_statement("pass"))

    return cst.FunctionDef(
      name=cst.Name("__init__"),
      params=cst.Parameters(params=params),
      body=cst.IndentedBlock(body=stmts),
    )

  def _build_forward(self, nodes: List[LogicalNode]) -> cst.FunctionDef:
    stmts: List[cst.BaseStatement] = []
    input_nodes = [n for n in nodes if n.kind == "Input"]
    input_arg_name = "x"
    if input_nodes:
      input_arg_name = input_nodes[0].metadata.get("name", input_nodes[0].id)

    current_var = input_arg_name
    var_map = {input_nodes[0].id: input_arg_name} if input_nodes else {}

    for node in nodes:
      if node.kind == "Input":
        continue
      if node.kind == "Output":
        # Check if output is fed by something specific
        stmts.append(cst.parse_statement(f"return {current_var}"))
        continue

      if self._is_stateful_layer(node):
        line = f"{current_var} = self.{node.id}({current_var})"
        stmts.append(cst.parse_statement(line))
      else:
        func_api = node.kind
        args_str = current_var
        if node.metadata:
          extra_args = self._format_args_from_metadata(node.metadata)
          if extra_args:
            args_str += f", {extra_args}"
        line = f"{current_var} = {func_api}({args_str})"
        stmts.append(cst.parse_statement(line))
      var_map[node.id] = current_var

    # Ensure return at end
    if not stmts or not m.matches(stmts[-1], m.Return()):
      stmts.append(cst.parse_statement(f"return {current_var}"))

    func_name = "forward"
    if self.framework in ["jax", "flax", "flax_nnx", "keras"]:
      func_name = "__call__" if self.framework != "keras" else "call"

    return cst.FunctionDef(
      name=cst.Name(func_name),
      params=cst.Parameters(
        params=[
          cst.Param(name=cst.Name("self")),
          cst.Param(name=cst.Name(input_arg_name)),
        ]
      ),
      body=cst.IndentedBlock(body=stmts),
    )

  def _is_stateful_layer(self, node: LogicalNode) -> bool:
    if node.kind in ["Input", "Output"]:
      return False
    if "." in node.kind and not node.kind.startswith("nn."):
      return False
    return True

  def _generate_layer_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
    kind = node.kind
    if "." not in kind:
      if self.framework == "torch":
        kind = f"nn.{kind}"
      elif self.framework in ["jax", "flax", "flax_nnx"]:
        kind = f"nnx.{kind}"
      elif self.framework == "keras":
        kind = f"keras.layers.{kind}"

    args_str = self._format_args_from_metadata(node.metadata)
    if self.framework in ["jax", "flax", "flax_nnx"]:
      if "rngs" not in args_str:
        suffix = ", rngs=rngs" if args_str else "rngs=rngs"
        args_str += suffix

    code = f"self.{node.id} = {kind}({args_str})"
    return cst.parse_statement(code)

  def _format_args_from_metadata(self, metadata: Dict[str, Any]) -> str:
    if not metadata:
      return ""
    args_list = []
    for key in sorted(metadata.keys()):
      val = str(metadata[key])
      if key.startswith("arg_"):
        args_list.append(val)
      else:
        args_list.append(f"{key}={val}")
    return ", ".join(args_list)
