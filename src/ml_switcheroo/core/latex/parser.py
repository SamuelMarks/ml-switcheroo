# src/ml_switcheroo/core/latex/parser.py

"""
LaTeX DSL Parser.

Parses MIDL LaTeX macros into a LibCST Module representing a Python AST.

The parser transforms the declarative LaTeX structure into an object-oriented
Python representation wrapped in a specific namespace (`midl`). This allows
downstream attributes and operations to be explicitly targeted by the rewriter.

Example Transformation:
    LaTeX: \Attribute{conv}{Conv2d}{k=3}
    Python: self.conv = midl.Conv2d(k=3)
"""

import re
from typing import List, Dict
import libcst as cst
from ml_switcheroo.core.latex.nodes import (
  MemoryNode,
  InputNode,
  ComputeNode,
  StateOpNode,
  ReturnNode,
)


class LatexParser:
  """
  Parses LaTeX source code containing MIDL macros into a Python CST.
  """

  _START_ENV = re.compile(r"\\begin\{DefModel\}\{(?P<name>\w+)\}")
  _ATTR_RE = re.compile(r"\\Attribute\{(?P<id>[\w\d_]+)\}\{(?P<type>[\w\d_\.]+)\}\{(?P<config>.*?)\}")
  _INPUT_RE = re.compile(r"\\Input\{(?P<name>[\w\d_]+)\}\{(?P<shape>.*?)\}")
  _OP_RE = re.compile(r"\\Op\{(?P<id>[\w\d_]+)\}\{(?P<type>[\w\d_\.]+)\}\{(?P<args>.*?)\}\{(?P<shape>.*?)\}")
  _STATE_OP_RE = re.compile(r"\\StateOp\{(?P<id>[\w\d_]+)\}\{(?P<attr>[\w\d_]+)\}\{(?P<args>.*?)\}\{(?P<shape>.*?)\}")
  _RETURN_RE = re.compile(r"\\Return\{(?P<id>[\w\d_]+)\}")

  def __init__(self, latex_source: str):
    self.source = latex_source

  def parse(self) -> cst.Module:
    """
    Parses the internally stored LaTeX source string.

    Returns:
        cst.Module: A LibCST Module containing the synthesized Python class.
    """
    model_name = "GeneratedModel"
    memory_nodes = []
    compute_nodes = []
    input_node = None
    return_node = None

    for line in self.source.splitlines():
      line = line.strip()
      if not line or line.startswith("%"):
        continue

      m = self._START_ENV.search(line)
      if m:
        model_name = m.group("name")
        continue

      m = self._ATTR_RE.search(line)
      if m:
        d = m.groupdict()
        memory_nodes.append(MemoryNode(d["id"], d["type"], self._parse_config_string(d["config"])))
        continue

      m = self._INPUT_RE.search(line)
      if m:
        input_node = InputNode(m.group("name"), m.group("shape"))
        continue

      m = self._OP_RE.search(line)
      if m:
        d = m.groupdict()
        compute_nodes.append(ComputeNode(d["id"], d["type"], self._parse_arg_list(d["args"]), d["shape"]))
        continue

      m = self._STATE_OP_RE.search(line)
      if m:
        d = m.groupdict()
        compute_nodes.append(StateOpNode(d["id"], d["attr"], self._parse_arg_list(d["args"]), d["shape"]))
        continue

      m = self._RETURN_RE.search(line)
      if m:
        return_node = ReturnNode(m.group("id"))
        continue

    class_def = self._synthesize_class(model_name, memory_nodes, input_node, compute_nodes, return_node)

    # Inject `import midl` at file level
    imports = cst.SimpleStatementLine([cst.Import(names=[cst.ImportAlias(name=cst.Name("midl"))])])
    return cst.Module(body=[imports, class_def])

  def _parse_config_string(self, s: str) -> Dict[str, str]:
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
    if not s.strip():
      return []
    return [a.strip() for a in s.split(",")]

  def _safe_value_node(self, val: str) -> cst.BaseExpression:
    """
    Safely converts a string value to a LibCST Expression Node.

    Handles:
    - Ellipsis (...)
    - Python Expressions (Integers, Floats, Math, Strings, Lists)
    - Fallback Identifiers
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
    # Note: This might create invalid ASTs if 'val' is not a valid identifier,
    # but provides a last-ditch effort for raw strings passed as args in macros.
    return cst.Name(clean_val)

  def _create_call(self, func_name: str, config: Dict = None, args_list: List = None) -> cst.Call:
    if "." in func_name:
      p = func_name.split(".")
      fn = cst.Name(p[0])
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

  def _synthesize_class(self, name, mem, inp, ops, ret) -> cst.ClassDef:
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
