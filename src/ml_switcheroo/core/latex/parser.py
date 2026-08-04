"""LaTeX DSL Parser.

Parses MIDL LaTeX macros into a LibCST Module representing a Python AST.
"""

from typing import Any, List, Dict, Optional, Union
import libcst as cst
from ml_switcheroo.core.latex.nodes import (
  MemoryNode,
  InputNode,
  ComputeNode,
  StateOpNode,
  ReturnNode,
)


class LatexParser:
  """Parse LaTeX source code containing MIDL macros into a Python CST.

  This class traverses a LaTeX source string, identifies MIDL custom macros
  such as `Attribute`, `Input`, `Op`, `StateOp`, and `Return` within a `DefModel`
  environment, and translates them into a syntactically correct Python class structure.
  """

  def __init__(self, latex_source: str):
    """Initialize the parser with LaTeX source code.

    Args:
        latex_source (str): The raw LaTeX source code string to parse.
    """
    self.source = latex_source

  def parse(self) -> cst.Module:
    """Parse the internally stored LaTeX source string.

    Iterates through the LaTeX source to identify environment and macro definitions,
    collects logical components, and synthesizes them into a LibCST Module containing
    the translated Python model class and its imports.

    Returns:
        cst.Module: A LibCST Module containing the synthesized Python class.
    """
    model_name = "GeneratedModel"
    memory_nodes = []
    compute_nodes: list[Any] = []
    input_node = None
    return_node = None

    pos = 0
    in_def_model = False

    while pos < len(self.source):
      if self.source.startswith(r"\begin{", pos):
        end = self.source.find("}", pos)
        name = self.source[pos + 7 : end]
        pos = end + 1
        if name == "DefModel":
          in_def_model = True
          # Parse model name argument if present
          if pos < len(self.source) and self.source[pos] == "{":
            pos += 1
            depth = 1
            start_arg = pos
            while pos < len(self.source) and depth > 0:
              if self.source[pos] == "{":
                depth += 1
              elif self.source[pos] == "}":
                depth -= 1
              pos += 1
            model_name = self.source[start_arg : pos - 1].strip()
        continue

      if self.source.startswith(r"\end{", pos):
        end = self.source.find("}", pos)
        name = self.source[pos + 5 : end]
        pos = end + 1
        if name == "DefModel":
          in_def_model = False
        continue

      if self.source.startswith(r"\%", pos):
        pos += 2
        continue

      if self.source[pos] == "%":
        end = self.source.find("\n", pos)
        pos = end if end != -1 else len(self.source)
        continue

      if self.source[pos] == "\\":
        pos += 1
        start = pos
        if pos < len(self.source) and not self.source[pos].isalpha():
          pos += 1
          macro_name = self.source[start:pos]
        else:
          while pos < len(self.source) and self.source[pos].isalpha():
            pos += 1
          macro_name = self.source[start:pos]

        args = []
        while pos < len(self.source):
          saved_pos = pos
          while pos < len(self.source) and self.source[pos] in (" ", "\t", "\n", "\r"):
            pos += 1

          if pos < len(self.source) and self.source[pos] == "{":
            pos += 1
            depth = 1
            start_arg = pos
            while pos < len(self.source) and depth > 0:
              if self.source[pos] == "{":
                depth += 1
              elif self.source[pos] == "}":
                depth -= 1
              pos += 1
            args.append(self.source[start_arg : pos - 1])
          elif pos < len(self.source) and self.source[pos] == "[":
            pos += 1
            depth = 1
            start_arg = pos
            while pos < len(self.source) and depth > 0:
              if self.source[pos] == "[":
                depth += 1
              elif self.source[pos] == "]":
                depth -= 1
              pos += 1
            args.append("[" + self.source[start_arg : pos - 1] + "]")
          else:
            pos = saved_pos
            break

        if in_def_model:
          if macro_name == "Attribute" and len(args) >= 3:
            memory_nodes.append(MemoryNode(args[0], args[1], self._parse_config_string(args[2])))
          elif macro_name == "Input" and len(args) >= 2:
            input_node = InputNode(args[0], args[1])
          elif macro_name == "Op" and len(args) >= 4:
            compute_nodes.append(ComputeNode(args[0], args[1], self._parse_arg_list(args[2]), args[3]))
          elif macro_name == "StateOp" and len(args) >= 4:
            compute_nodes.append(StateOpNode(args[0], args[1], self._parse_arg_list(args[2]), args[3]))
          elif macro_name == "Return" and len(args) >= 1:
            return_node = ReturnNode(args[0])
        continue

      pos += 1

    class_def = self._synthesize_class(model_name, memory_nodes, input_node, compute_nodes, return_node)

    # Inject `import midl` at file level
    imports = cst.SimpleStatementLine([cst.Import(names=[cst.ImportAlias(name=cst.Name("midl"))])])
    return cst.Module(body=[imports, class_def])

  def _parse_config_string(self, s: str) -> Dict[str, str]:
    """Parse 'key=value, key2=value2' strings.

    Args:
        s (str): A comma-separated string of configuration arguments.

    Returns:
        Dict[str, str]: A dictionary of key-value pairs representing configuration properties.
    """
    if not s.strip():
      return {}
    res = {}
    for p in s.split(","):
      if "=" in p:
        k, v = p.split("=", 1)
        res[k.strip()] = v.strip()
      else:
        res[f"arg_{len(res)}"] = p.strip()
    return res

  def _parse_arg_list(self, s: str) -> List[str]:
    """Parse comma-separated arguments logic.

    Args:
        s (str): A string representing a comma-separated list of arguments.

    Returns:
        List[str]: A list of cleaned/trimmed argument strings.
    """
    if not s.strip():
      return []
    return [a.strip() for a in s.split(",")]

  def _safe_value_node(self, val: str) -> cst.BaseExpression:
    """Safely converts a string value to a LibCST Expression Node.

    Handles:
    - Ellipsis (...)
    - Python Expressions (Integers, Floats, Math, Strings, Lists)
    - Fallback Identifiers

    Args:
        val (str): The raw string value to convert.

    Returns:
        cst.BaseExpression: A LibCST expression node matching the input string value.
    """
    clean_val = val.strip()

    # 1. Ellipsis Fallback
    if clean_val == "...":
      return cst.Ellipsis()

    # 2. Attempt Expression Parsing
    try:
      return cst.parse_expression(clean_val)
    except cst.ParserSyntaxError:
      pass

    # 3. Fallback to Identifier (Name)
    return cst.Name(clean_val)

  def _create_call(
    self, func_name: str, config: Optional[Dict[str, str]] = None, args_list: Optional[List[str]] = None
  ) -> cst.Call:
    """Construct a CST Call node from config and arguments.

    Args:
        func_name (str): The name or path of the function/class to call.
        config (Optional[Dict[str, str]]): Configuration keyword parameters to include.
        args_list (Optional[List[str]]): A list of positional or keyword argument strings.

    Returns:
        cst.Call: A synthesized LibCST Call node with the appropriate arguments and tight equals formatting.
    """
    if "." in func_name:
      p = func_name.split(".")
      fn: cst.BaseExpression = cst.Name(p[0])
      for x in p[1:]:
        fn = cst.Attribute(value=fn, attr=cst.Name(x))
    else:
      fn = cst.Name(func_name)

    # AssignEqual with empty whitespace around '='
    # to match tests expecting "key=val" instead of "key = val"
    eq = cst.AssignEqual(
      whitespace_before=cst.SimpleWhitespace(""),
      whitespace_after=cst.SimpleWhitespace(""),
    )

    args = []
    if config:
      for k, v in config.items():
        if k.startswith("arg_"):
          args.append(cst.Arg(value=self._safe_value_node(v)))
        else:
          # Use tight equality
          args.append(
            cst.Arg(
              keyword=cst.Name(k),
              value=self._safe_value_node(v),
              equal=eq,
            )
          )

    if args_list:
      for item in args_list:
        if "=" in item:
          k, v = item.split("=", 1)
          # Use tight equality for string-parsed args
          args.append(
            cst.Arg(
              keyword=cst.Name(k.strip()),
              value=self._safe_value_node(v.strip()),
              equal=eq,
            )
          )
        else:
          args.append(cst.Arg(value=self._safe_value_node(item)))

    return cst.Call(func=fn, args=args)

  def _synthesize_class(
    self,
    name: str,
    mem: List[MemoryNode],
    inp: Optional[InputNode],
    ops: List[Union[ComputeNode, StateOpNode]],
    ret: Optional[ReturnNode],
  ) -> cst.ClassDef:
    """Combine logical components into a Python Class AST.

    Args:
        name (str): The name of the synthesized model class.
        mem (List[MemoryNode]): State/attribute definitions mapped to attributes.
        inp (Optional[InputNode]): Input specification mapping input identifiers.
        ops (List[Union[ComputeNode, StateOpNode]]): The ordered sequence of operations.
        ret (Optional[ReturnNode]): The Return statement representing output results.

    Returns:
        cst.ClassDef: A LibCST ClassDef node representing the complete generated class.
    """
    init_body = [cst.SimpleStatementLine([cst.Expr(cst.parse_expression("super().__init__()"))])]
    for m in mem:
      tgt = cst.Attribute(value=cst.Name("self"), attr=cst.Name(m.node_id))
      # Prefix ops with 'midl.' to ensure they are picked up by the Semantic Rewriter
      # E.g. Conv2d -> midl.Conv2d
      op_target = f"midl.{m.op_type}"
      init_body.append(
        cst.SimpleStatementLine(
          [
            cst.Assign(
              targets=[cst.AssignTarget(tgt)],
              value=self._create_call(op_target, config=m.config),
            )
          ]
        )
      )

    fwd_body = []
    for op in ops:
      lhs = cst.AssignTarget(cst.Name(op.node_id))
      if isinstance(op, ComputeNode):
        # Stateless op usage: prefix with 'midl.'
        # E.g. Flatten -> midl.Flatten
        op_target = f"midl.{op.op_type}"
        rhs = self._create_call(op_target, args_list=op.args)
      elif isinstance(op, StateOpNode):
        # Stateful op usage: call self.attribute
        fname = f"self.{op.attribute_id}"
        rhs = self._create_call(fname, args_list=op.args)
      else:
        # Fallback, though should not be reachable given loop source
        rhs = cst.Name("None")

      fwd_body.append(cst.SimpleStatementLine([cst.Assign(targets=[lhs], value=rhs)]))

    if ret:
      fwd_body.append(cst.SimpleStatementLine([cst.Return(cst.Name(ret.target_id))]))
    else:
      fwd_body.append(cst.SimpleStatementLine([cst.Pass()]))

    in_name = inp.name if inp else "x"

    # Base class: midl.Module
    base_class = cst.Arg(value=cst.Attribute(value=cst.Name("midl"), attr=cst.Name("Module")))

    return cst.ClassDef(
      name=cst.Name(name),
      bases=[base_class],
      body=cst.IndentedBlock(
        body=[
          cst.FunctionDef(
            name=cst.Name("__init__"),
            params=cst.Parameters(params=[cst.Param(cst.Name("self"))]),
            body=cst.IndentedBlock(init_body),
          ),
          cst.FunctionDef(
            name=cst.Name("forward"),
            params=cst.Parameters(
              params=[
                cst.Param(cst.Name("self")),
                cst.Param(cst.Name(in_name)),
              ]
            ),
            body=cst.IndentedBlock(fwd_body),
          ),
        ]
      ),
    )
