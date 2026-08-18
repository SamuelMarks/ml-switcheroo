"""HTML Parser.

Parses the HTML DSL structure using a formal Lark grammar to construct an HTML CST (`HtmlDocument`),
and extracts high-level model logic into a Python LibCST Module.
"""

from typing import Any, List, Tuple, Optional

import libcst as cst
from html.parser import HTMLParser as BaseHTMLParser

from ml_switcheroo.core.html.nodes import HtmlDocument, TagNode, TextNode, CommentNode, AttributeNode, HtmlNode


class InternalHtmlParser(BaseHTMLParser):
  """Builds the HtmlDocument CST from SAX-like events."""

  def __init__(self) -> None:
    """Initialize the parser."""
    super().__init__()
    self.root_children: List[HtmlNode] = []
    self.stack: List[TagNode] = []

  def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
    """Handle a start tag in the HTML document.

    Args:
        tag (str): The name of the HTML tag.
        attrs (list of tuple): List of (name, value) pairs representing the attributes.
    """
    attributes = []
    for k, v in attrs:
      if v is None:
        attributes.append(AttributeNode(name=k, value=None, quote_style=""))
      else:
        attributes.append(AttributeNode(name=k, value=v, quote_style='"'))

    node = TagNode(name=tag, attributes=attributes, children=[])

    void_elements = {
      "area",
      "base",
      "br",
      "col",
      "embed",
      "hr",
      "img",
      "input",
      "link",
      "meta",
      "param",
      "source",
      "track",
      "wbr",
    }
    if tag in void_elements:
      node.self_closing = True
      self._append_node(node)
    else:
      self.stack.append(node)

  def handle_endtag(self, tag: str) -> None:
    """Handle an end tag in the HTML document.

    Args:
        tag (str): The name of the HTML tag being closed.
    """
    for i in range(len(self.stack) - 1, -1, -1):
      if self.stack[i].name == tag:
        closed_nodes = self.stack[i:]
        self.stack = self.stack[:i]

        node = closed_nodes[0]
        for unclosed in closed_nodes[1:]:
          node.children.append(unclosed)

        self._append_node(node)
        break

  def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
    """Handle a self-closing tag in the HTML document.

    Args:
        tag (str): The name of the HTML tag.
        attrs (list of tuple): List of (name, value) pairs representing the attributes.
    """
    attributes = []
    for k, v in attrs:
      if v is None:
        attributes.append(AttributeNode(name=k, value=None, quote_style=""))
      else:
        attributes.append(AttributeNode(name=k, value=v, quote_style='"'))
    node = TagNode(name=tag, attributes=attributes, self_closing=True)
    self._append_node(node)

  def handle_data(self, data: str) -> None:
    """Handle text data in the HTML document.

    Args:
        data (str): The text content encountered.
    """
    self._append_node(TextNode(content=data))

  def handle_comment(self, data: str) -> None:
    """Handle an HTML comment.

    Args:
        data (str): The content of the HTML comment.
    """
    self._append_node(CommentNode(content=data))

  def handle_decl(self, decl: str) -> None:
    """Handle an HTML declaration.

    Args:
        decl (str): The declaration string (e.g., "DOCTYPE html").
    """
    node = TagNode(name="!" + decl, self_closing=True)
    self._append_node(node)

  def _append_node(self, node: HtmlNode) -> None:
    """Append a node.

    Args:
        node (HtmlNode): The node to append, either as a child of the current stack top
            or directly to the root children if the stack is empty.
    """
    if self.stack:
      self.stack[-1].children.append(node)
    else:
      self.root_children.append(node)


class GridExtractor:
  """Extracts logical operations from an HtmlDocument CST."""

  def __init__(self) -> None:
    """Initialize the GridExtractor."""
    self.model_name = "Model"
    self.attrs: List[Tuple[str, str, str]] = []
    self.ops: List[Tuple[str, str]] = []

  def extract(self, doc: HtmlDocument) -> None:
    """Extract the model properties.

    Args:
        doc (HtmlDocument): The HTML document CST from which to extract properties.
    """
    self.model_name = doc.model_name
    self._walk(doc.children)
    if not self.model_name or self.model_name == "Model":
      self.model_name = "Model"

  def _walk(self, nodes: List[HtmlNode]) -> None:
    """Recursively walk the DOM to extract nodes.

    Args:
        nodes (List[HtmlNode]): A list of nodes to recursively traverse.
    """
    for node in nodes:
      if isinstance(node, TagNode):
        if node.name == "h3":
          full_text = ""
          for c in node.children:
            if isinstance(c, TextNode):
              full_text += c.content
          if "Model:" in full_text:
            self.model_name = full_text.replace("Model:", "").strip()
        elif node.name == "div":
          classes = ""
          for attr in node.attributes:
            if attr.name == "class" and attr.value:
              classes = attr.value
          if "box" in classes:
            self._process_box(node, classes)
        self._walk(node.children)

  def _process_box(self, node: TagNode, classes: str) -> None:
    """Extract information from a grid box.

    Args:
        node (TagNode): The CST TagNode representing the grid box.
        classes (str): CSS classes associated with the grid box div.
    """
    header_txt = ""
    code_txt = ""

    for child in node.children:
      if isinstance(child, TagNode):
        if child.name == "span":
          for c in child.children:
            if isinstance(c, TextNode):
              header_txt += c.content
        elif child.name == "code":
          for c in child.children:
            if isinstance(c, TextNode):
              code_txt += c.content

    header_txt = header_txt.strip()
    code_txt = code_txt.strip()
    classes_list = classes.strip().split()

    if "r" in classes_list:
      # Red boxes are attributes (Layers)
      if ":" in header_txt:
        name, kind = header_txt.split(":", 1)
        self.attrs.append((name.strip(), kind.strip(), code_txt))
      else:
        self.attrs.append((header_txt, "Unknown", code_txt))

    elif "b" in classes_list:
      # Blue boxes are Call operations
      op = header_txt
      # Only handle operations, not headers
      if not op.startswith("Call"):
        self.ops.append((op, code_txt))
      elif op.startswith("Call ("):
        # Handle stateful calls: Header is "Call (conv)"
        # We store full op name to parse later
        self.ops.append((op, code_txt))


class HtmlParser:
  """Facade for parsing HTML strings into LibCST modules using the CST parser."""

  def __init__(self, source: str) -> None:
    """Initialize the HtmlParser with raw HTML source string.

    Args:
        source (str): The raw HTML source string containing the DSL elements.
    """
    self.source = source

  def parse_cst(self) -> HtmlDocument:
    """Parse the HTML string into an HtmlDocument.

    Returns:
        HtmlDocument: The parsed HTML document CST representation.
    """
    parser = InternalHtmlParser()
    parser.feed(self.source)

    while parser.stack:
      unclosed = parser.stack.pop()
      if parser.stack:
        parser.stack[-1].children.append(unclosed)
      else:
        parser.root_children.append(unclosed)

    model_name = "Model"

    def _find_model(nodes: List[HtmlNode]) -> None:
      """Find the model name.

      Args:
          nodes (List[HtmlNode]): List of nodes to search for model headers.
      """
      nonlocal model_name
      for n in nodes:
        if isinstance(n, TagNode):
          if n.name == "h3":
            full_text = "".join(c.content for c in n.children if isinstance(c, TextNode))
            if "Model:" in full_text:
              model_name = full_text.replace("Model:", "").strip()
          _find_model(n.children)

    _find_model(parser.root_children)

    return HtmlDocument(model_name=model_name, children=parser.root_children)

  def parse(self) -> cst.Module:
    """Parse the HTML string and construct a Python LibCST Module representing the model.

    Returns:
        cst.Module: The parsed LibCST Module.
    """
    doc = self.parse_cst()

    p = GridExtractor()
    p.extract(doc)

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

  def _create_dotted(self, name: Any) -> Any:
    """Create a dotted LibCST Name or Attribute node from a dot-separated string.

    Args:
        name (Any): The dotted name as a string or other representable type.

    Returns:
        Any: The constructed LibCST Attribute or Name node.
    """
    parts = name.split(".")
    node = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(node, cst.Name(p))  # type: ignore
    return node

  def _create_call(self, func_name: Any, config_str: Any = None) -> Any:
    """Create a LibCST Call node.

    Args:
        func_name (Any): The name of the function to be called.
        config_str (Any, optional): The string representing the function configuration arguments.

    Returns:
        Any: The constructed LibCST Call node.
    """
    args = []
    if config_str:
      args = self._parse_args_str(config_str)
    return cst.Call(func=self._create_dotted(func_name), args=args)

  def _parse_args_str(self, s: str) -> List[cst.Arg]:
    """Parse key=val, key2=val2 string into CST Args.

    Args:
        s (str): The raw comma-separated arguments string.

    Returns:
        List[cst.Arg]: A list of LibCST Arg nodes.
    """
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

  def _safe_val(self, v: Any) -> Any:
    """Safely parse a string value into a LibCST expression node.

    Args:
        v (Any): The string or object representing the value to parse.

    Returns:
        Any: The parsed LibCST expression node, or SimpleString if parsing fails.
    """
    try:
      return cst.parse_expression(v)
    except Exception:
      return cst.SimpleString(f"'{v}'")
