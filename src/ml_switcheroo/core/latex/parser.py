"""LaTeX DSL Parser.

Parses MIDL LaTeX macros into a LibCST Module representing a Python AST,
built atop the shared `BaseTokenizer` and `RecursiveDescentParser` infrastructure
in `ml_switcheroo.core.markup.parser`.
"""

import libcst as cst

from ml_switcheroo.core.latex.nodes import ComputeNode, InputNode, MemoryNode, ReturnNode, StateOpNode
from ml_switcheroo.core.markup.parser import BaseTokenizer, RecursiveDescentParser, Token

_LATEX_TOKEN_SPEC: list[tuple[str, str]] = [
  ("ESCAPED_PERCENT", r"\\%"),
  ("COMMENT", r"%[^\n]*"),
  ("BEGIN", r"\\begin\{[^}]*\}"),
  ("MACRO", r"\\(?:[a-zA-Z]+|[^a-zA-Z\s])"),
  ("LBRACE", r"\{"),
  ("RBRACE", r"\}"),
  ("LBRACKET", r"\["),
  ("RBRACKET", r"\]"),
  ("SKIP", r"\s+"),
  ("TEXT", r"[^\\\{\}\[\]%\s]+"),
]


class _LatexRecursiveDescentParser(RecursiveDescentParser):
  """Recursive descent parser over tokenized MIDL LaTeX streams."""

  def _consume_balanced(self, open_tok: str, close_tok: str) -> str:
    """Consume a balanced ``{...}`` or ``[...]`` group and return its raw inner text.

    Args:
        open_tok: Opening token type (``"LBRACE"`` or ``"LBRACKET"``).
        close_tok: Closing token type (``"RBRACE"`` or ``"RBRACKET"``).

    Returns:
        Concatenated inner string content between the balanced delimiters.
    """
    self.expect(open_tok)
    depth = 1
    parts: list[str] = []
    while self.current is not None and depth > 0:
      tok = self.advance()
      assert tok is not None
      if tok.type == open_tok:
        depth += 1
        parts.append(tok.value)
      elif tok.type == close_tok:
        depth -= 1
        if depth > 0:
          parts.append(tok.value)
      else:
        parts.append(tok.value)
    return "".join(parts)

  def parse_document(
    self,
  ) -> tuple[
    str,
    list[MemoryNode],
    InputNode | None,
    list[ComputeNode | StateOpNode],
    ReturnNode | None,
    list[tuple[str, list[str]]],
  ]:
    """Parse the token stream and extract `DefModel` macro components.

    Returns:
        Tuple of `(model_name, raw_attributes, raw_input, raw_ops, raw_return, raw_macros)`.
    """
    model_name = "GeneratedModel"
    raw_macros: list[tuple[str, list[str]]] = []

    while self.current is not None:
      tok = self.current
      if tok.type == "BEGIN":
        self.advance()
        env_name = tok.value[len(r"\begin{") : -1]
        cur_tok: Token | None = self.current
        if env_name == "DefModel" and cur_tok is not None and cur_tok.type == "LBRACE":
          model_name = self._consume_balanced("LBRACE", "RBRACE").strip()
      elif tok.type == "MACRO":
        self.advance()
        macro_name = tok.value[1:]
        args: list[str] = []
        while True:
          next_tok: Token | None = self.current
          if next_tok is None or next_tok.type not in ("LBRACE", "LBRACKET"):
            break
          if next_tok.type == "LBRACE":
            args.append(self._consume_balanced("LBRACE", "RBRACE"))
          else:
            args.append("[" + self._consume_balanced("LBRACKET", "RBRACKET") + "]")
        raw_macros.append((macro_name, args))
      else:
        self.advance()

    return model_name, [], None, [], None, raw_macros


class LatexParser:
  """Parse LaTeX source code containing MIDL macros into a Python CST.

  This class tokenizes and parses a LaTeX source string using `BaseTokenizer`
  and `RecursiveDescentParser`, identifies MIDL custom macros such as `Attribute`,
  `Input`, `Op`, `StateOp`, and `Return` within a `DefModel` environment, and
  translates them into a syntactically correct Python class structure.
  """

  def __init__(self, latex_source: str) -> None:
    """Initialize the parser with LaTeX source code.

    Args:
        latex_source (str): The raw LaTeX source code string to parse.
    """
    self.source = latex_source
    self._tokenizer = BaseTokenizer(_LATEX_TOKEN_SPEC)

  def parse(self) -> cst.Module:
    """Parse the internally stored LaTeX source string.

    Tokenizes and parses the LaTeX source to identify environment and macro definitions,
    collects logical components, and synthesizes them into a LibCST Module containing
    the translated Python model class and its imports.

    Returns:
        cst.Module: A LibCST Module containing the synthesized Python class.
    """
    tokens: list[Token] = self._tokenizer.tokenize(self.source)
    rd_parser = _LatexRecursiveDescentParser(tokens)
    model_name, _, _, _, _, raw_macros = rd_parser.parse_document()

    memory_nodes: list[MemoryNode] = []
    compute_nodes: list[ComputeNode | StateOpNode] = []
    input_node: InputNode | None = None
    return_node: ReturnNode | None = None

    for macro_name, args in raw_macros:
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

    class_def = self._synthesize_class(model_name, memory_nodes, input_node, compute_nodes, return_node)

    # Inject `import midl` at file level
    imports = cst.SimpleStatementLine([cst.Import(names=[cst.ImportAlias(name=cst.Name("midl"))])])
    return cst.Module(body=[imports, class_def])

  def _parse_config_string(self, s: str) -> dict[str, str]:
    """Parse 'key=value, key2=value2' strings.

    Args:
        s (str): A comma-separated string of configuration arguments.

    Returns:
        Dict[str, str]: A dictionary of key-value pairs representing configuration properties.
    """
    if not s.strip():
      return {}
    res: dict[str, str] = {}
    for p in s.split(","):
      if "=" in p:
        k, v = p.split("=", 1)
        res[k.strip()] = v.strip()
      else:
        res[f"arg_{len(res)}"] = p.strip()
    return res

  def _parse_arg_list(self, s: str) -> list[str]:
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
    """Safely convert a string value to a LibCST Expression Node.

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

    if clean_val == "...":
      return cst.Ellipsis()

    try:
      return cst.parse_expression(clean_val)
    except cst.ParserSyntaxError:
      pass

    return cst.Name(clean_val)

  def _create_call(
    self,
    func_name: str,
    config: dict[str, str] | None = None,
    args_list: list[str] | None = None,
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
    mem: list[MemoryNode],
    inp: InputNode | None,
    ops: list[ComputeNode | StateOpNode],
    ret: ReturnNode | None,
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
        op_target = f"midl.{op.op_type}"
        rhs: cst.BaseExpression = self._create_call(op_target, args_list=op.args)
      elif hasattr(op, "attribute_id"):
        fname = f"self.{op.attribute_id}"
        rhs = self._create_call(fname, args_list=op.args)
      else:
        rhs = cst.Name("None")

      fwd_body.append(cst.SimpleStatementLine([cst.Assign(targets=[lhs], value=rhs)]))

    if ret:
      fwd_body.append(cst.SimpleStatementLine([cst.Return(cst.Name(ret.target_id))]))
    else:
      fwd_body.append(cst.SimpleStatementLine([cst.Pass()]))

    in_name = inp.name if inp else "x"
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
