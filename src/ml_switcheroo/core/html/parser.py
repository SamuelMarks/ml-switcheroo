"""
HTML Parser.

Parses the HTML DSL structure using standard library `html.parser`.
Reconstructs a Python LibCST Module representing the logic described by the visual grid.
"""

import libcst as cst
from html.parser import HTMLParser
from typing import List, Tuple, Optional


class GridParser(HTMLParser):
  """
  HTML Parser callback handler.
  Extracts high-level model components (Name, Attributes, Operations) from the DOM stream.
  """

  def __init__(self) -> None:
    super().__init__()
    self.in_box = False
    self.current_class = ""
    self.in_header_txt = False
    self.in_code = False
    self.model_name = "Model"
    self.attrs: List[Tuple[str, str, str]] = []  # (name, kind, config_str)
    self.ops: List[Tuple[str, str]] = []  # (op_name, args_str)
    self._buf_header = ""
    self._buf_code = ""

  def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
    attrs_dict = dict((k, v) for k, v in attrs if v is not None)
    cls = attrs_dict.get("class", "")

    if tag == "div" and "box" in cls:
      self.in_box = True
      self.current_class = cls
      self._buf_header = ""
      self._buf_code = ""

    if tag == "span" and "header-txt" in cls:
      self.in_header_txt = True

    if tag == "code":
      self.in_code = True

  def handle_endtag(self, tag: str) -> None:
    if tag == "div" and self.in_box:
      self.in_box = False
      self._process_box()
      self.current_class = ""

    if tag == "span":
      self.in_header_txt = False

    if tag == "code":
      self.in_code = False

  def handle_data(self, data: str) -> None:
    if self.in_header_txt:
      self._buf_header += data
    if self.in_code:
      self._buf_code += data
    if "Model:" in data:
      clean = data.replace("Model:", "").strip()
      if clean:
        self.model_name = clean

  def _process_box(self) -> None:
    txt = self._buf_header.strip()
    code = self._buf_code.strip()
    classes = self.current_class.strip().split()

    if "r" in classes:
      # Red boxes are attributes (Layers)
      if ":" in txt:
        name, kind = txt.split(":", 1)
        self.attrs.append((name.strip(), kind.strip(), code))
      else:
        self.attrs.append((txt, "Unknown", code))

    elif "b" in classes:
      # Blue boxes are Call operations
      op = txt
      # Only handle operations, not headers
      if not op.startswith("Call"):
        self.ops.append((op, code))
      elif op.startswith("Call ("):
        # Handle stateful calls: Header is "Call (conv)"
        # We store full op name to parse later
        self.ops.append((op, code))


class HtmlParser:
  """
  Facade for parsing HTML strings into LibCST modules.
  """

  def __init__(self, source: str) -> None:
    self.source = source

  def parse(self) -> cst.Module:
    p = GridParser()
    p.feed(self.source)

    # 1. Imports
    import_stmt = cst.SimpleStatementLine(
      [cst.Import(names=[cst.ImportAlias(name=cst.Name("html_dsl"), asname=cst.AsName(cst.Name("dsl")))])]
    )

    # 2. Build __init__
    init_stmts = []
    if p.attrs:
      init_stmts.append(cst.SimpleStatementLine([cst.Expr(cst.parse_expression("super().__init__()"))]))

    for name, kind, cfg in p.attrs:
      # Clean formatting logic: args like "args: x" means empty config for attribute
      config_str = cfg
      if cfg.startswith("args:"):
        config_str = ""

      target_api_class = f"dsl.{kind}"

      # Safely construct the RHS expression
      if not config_str:
        # Fallback for empty config
        rhs = cst.Call(func=self._create_dotted(target_api_class), args=[])
      else:
        # Use robustness helper
        rhs = self._create_call(target_api_class, config_str=config_str)

      target = cst.Attribute(value=cst.Name("self"), attr=cst.Name(name))
      init_stmts.append(cst.SimpleStatementLine([cst.Assign(targets=[cst.AssignTarget(target)], value=rhs)]))

    if not init_stmts:
      # Pass if empty
      init_stmts.append(cst.SimpleStatementLine([cst.Pass()]))

    # 3. Build forward
    fwd_stmts = []
    last_var = "x"

    for op_name, args in p.ops:
      # Parse args string: "args: x"
      clean_args = args.replace("args:", "").strip()

      if op_name.startswith("Call (") and op_name.endswith(")"):
        attr = op_name[6:-1]
        call_var = last_var
        out_var = f"{attr}_out"

        final_args_node = [cst.Arg(cst.Name(call_var))]

        rhs = cst.Call(func=cst.Attribute(value=cst.Name("self"), attr=cst.Name(attr)), args=final_args_node)
        fwd_stmts.append(cst.SimpleStatementLine([cst.Assign(targets=[cst.AssignTarget(cst.Name(out_var))], value=rhs)]))
        last_var = out_var
      else:
        # Functional Op
        target_api = f"dsl.{op_name}"

        # Default recursive logic: use last_var as first argument
        final_args_node = [cst.Arg(cst.Name(last_var))]

        if clean_args and clean_args != "x":
          # If additional args provided (not just x reference), parse them
          extra_args = self._parse_args_str(clean_args)
          final_args_node.extend(extra_args)

        out_var = f"{op_name.lower()}_out"

        rhs = cst.Call(func=self._create_dotted(target_api), args=final_args_node)
        fwd_stmts.append(cst.SimpleStatementLine([cst.Assign(targets=[cst.AssignTarget(cst.Name(out_var))], value=rhs)]))
        last_var = out_var

    fwd_stmts.append(cst.SimpleStatementLine([cst.Return(cst.Name(last_var))]))

    # 4. Construct Class
    class_def = cst.ClassDef(
      name=cst.Name(p.model_name),
      bases=[cst.Arg(cst.Attribute(cst.Name("dsl"), cst.Name("Module")))],
      body=cst.IndentedBlock(
        [
          cst.FunctionDef(
            name=cst.Name("__init__"),
            params=cst.Parameters(params=[cst.Param(cst.Name("self"))]),
            body=cst.IndentedBlock(init_stmts),
          ),
          cst.FunctionDef(
            name=cst.Name("forward"),
            params=cst.Parameters(params=[cst.Param(cst.Name("self")), cst.Param(cst.Name("x"))]),
            body=cst.IndentedBlock(fwd_stmts),
          ),
        ]
      ),
    )

    return cst.Module(body=[import_stmt, class_def])

  def _create_dotted(self, name):
    parts = name.split(".")
    node = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(node, cst.Name(p))
    return node

  def _create_call(self, func_name, config_str=None):
    args = []
    if config_str:
      args = self._parse_args_str(config_str)
    return cst.Call(func=self._create_dotted(func_name), args=args)

  def _parse_args_str(self, s: str) -> List[cst.Arg]:
    """Parses key=val, key2=val2 string into CST Args."""
    if not s:
      return []
    args = []
    parts = s.split(",")
    for p in parts:
      if "=" in p:
        k, v = p.split("=", 1)
        val_node = self._safe_val(v.strip())
        args.append(
          cst.Arg(
            keyword=cst.Name(k.strip()),
            value=val_node,
            equal=cst.AssignEqual(cst.SimpleWhitespace(""), cst.SimpleWhitespace("")),
          )
        )
      else:
        args.append(cst.Arg(self._safe_val(p.strip())))
    return args

  def _safe_val(self, v):
    try:
      return cst.parse_expression(v)
    except:
      return cst.SimpleString(f"'{v}'")
