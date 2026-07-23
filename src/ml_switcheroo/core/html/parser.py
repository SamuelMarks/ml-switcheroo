"""HTML Parser.

Parses the HTML DSL structure using a formal Lark grammar to construct an HTML CST (`HtmlDocument`),
and extracts high-level model logic into a Python LibCST Module.
"""

import os
from typing import Any, List, Tuple

import libcst as cst
from lark import Lark, Transformer, Token

from ml_switcheroo.core.html.nodes import HtmlDocument, TagNode, TextNode, CommentNode, AttributeNode, HtmlNode


class HtmlTransformer(Transformer[Token, Any]):
  """Transforms a parsed HTML AST into HTML CST components."""

  def start(self, children: List[Any]) -> HtmlDocument:
    """Process the start rule."""
    elements = []
    for c in children:
      if isinstance(c, list):
        elements.extend(c)
      elif isinstance(c, (TagNode, TextNode, CommentNode)):  # pragma: no cover
        elements.append(c)  # pragma: no cover

    # Find model name from h3. In flat sequence h3 is a TagNode with children.
    model_name = "Model"
    for el in elements:
      if isinstance(el, TagNode) and el.name == "h3":
        for child in el.children:
          if isinstance(child, TextNode) and "Model:" in child.content:  # pragma: no cover
            model_name = child.content.replace("Model:", "").strip()  # pragma: no cover

    return HtmlDocument(model_name=model_name, children=elements)

  def element(self, children: List[Any]) -> List[Any]:
    """Process an element rule."""
    return children

  def text(self, children: List[Token]) -> TextNode:
    """Process a text rule."""
    return TextNode(content="".join(str(c) for c in children))

  def comment(self, children: List[Token]) -> CommentNode:
    """Process a comment rule."""
    content = "".join(str(c) for c in children)
    # Strip <!-- and -->
    if content.startswith("<!--") and content.endswith("-->"):
      content = content[4:-3]
    return CommentNode(content=content)

  def doctype(self, children: List[Any]) -> TagNode:
    """Process a doctype rule."""
    return TagNode(name="!DOCTYPE", self_closing=True, attributes=[AttributeNode(name="html")])  # pragma: no cover

  def tag_start(self, children: List[Any]) -> TagNode:
    """Process a tag_start rule."""
    name = ""
    attrs = []
    for c in children:
      if isinstance(c, Token) and c.type == "IDENTIFIER" and not name:
        name = str(c)
      elif isinstance(c, AttributeNode):
        attrs.append(c)
    return TagNode(name=name, attributes=attrs)

  def tag_end(self, children: List[Any]) -> str:
    """Process a tag_end rule."""
    for c in children:
      if isinstance(c, Token) and c.type == "IDENTIFIER":
        return str(c)
    return ""  # pragma: no cover

  def tag_self_closing(self, children: List[Any]) -> TagNode:
    """Process a tag_self_closing rule."""
    name = ""  # pragma: no cover
    attrs = []  # pragma: no cover
    for c in children:  # pragma: no cover
      if isinstance(c, Token) and c.type == "IDENTIFIER" and not name:  # pragma: no cover
        name = str(c)  # pragma: no cover
      elif isinstance(c, AttributeNode):  # pragma: no cover
        attrs.append(c)  # pragma: no cover
    return TagNode(name=name, attributes=attrs, self_closing=True)  # pragma: no cover

  def attribute(self, children: List[Any]) -> AttributeNode:
    """Process an attribute rule."""
    name = ""
    val = None
    quote = ""
    for c in children:
      if isinstance(c, Token) and c.type == "IDENTIFIER" and not name:
        name = str(c)
      elif isinstance(c, Token) and c.type == "STRING":
        s = str(c)
        if s.startswith('"') and s.endswith('"'):
          val = s[1:-1]
          quote = '"'
        elif s.startswith("'") and s.endswith("'"):  # pragma: no cover
          val = s[1:-1]  # pragma: no cover
          quote = "'"  # pragma: no cover
        else:
          val = s  # pragma: no cover
      elif isinstance(c, Token) and c.type == "IDENTIFIER" and name:
        val = str(c)  # pragma: no cover
    return AttributeNode(name=name, value=val, quote_style=quote)

  def script_style(self, children: List[Token]) -> TagNode:
    """Process a script_style rule."""
    content = "".join(str(c) for c in children)  # pragma: no cover
    # Simple extraction
    name = "style" if "<style" in content else "script"  # pragma: no cover
    return TagNode(name=name, children=[TextNode(content=content)])  # pragma: no cover

  def ws(self, children: List[Token]) -> TextNode:
    """Process a ws rule."""
    return TextNode(content="".join(str(c) for c in children))


class GridExtractor:
  """Extracts logical operations from an HtmlDocument CST."""

  def __init__(self) -> None:
    """Initialize the GridExtractor."""
    self.model_name = "Model"
    self.attrs: List[Tuple[str, str, str]] = []
    self.ops: List[Tuple[str, str]] = []

  def extract(self, doc: HtmlDocument) -> None:
    """Extract the model properties."""
    self.model_name = doc.model_name
    self._walk(doc.children)
    if not self.model_name or self.model_name == "Model":
      self.model_name = "Model"  # pragma: no cover

  def _walk(self, nodes: List[HtmlNode]) -> None:
    """Recursively walk the DOM to extract nodes."""
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
    """Extract information from a grid box."""
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
        self.attrs.append((header_txt, "Unknown", code_txt))  # pragma: no cover

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
    """Execute implementation detail."""
    self.source = source
    grammar_path = os.path.join(os.path.dirname(__file__), "grammar.lark")
    with open(grammar_path, "r", encoding="utf-8") as f:
      self.grammar = f.read()
    self.parser = Lark(self.grammar, start="start", parser="earley")

  def parse_cst(self) -> HtmlDocument:
    """Parse the HTML string into an HtmlDocument."""
    tree = self.parser.parse(self.source)
    transformer = HtmlTransformer()

    # We need a custom walk to build the hierarchy for TagNode since the grammar
    # produces a flat stream of tag_start and tag_end for flexibility with malformed HTML.
    from typing import cast

    flat_elements = cast(HtmlDocument, transformer.transform(tree))

    stack = []
    root_children = []

    for el in flat_elements.children:
      if (
        isinstance(el, TagNode)
        and not el.self_closing
        and el.name not in ["script", "style", "br", "hr", "img", "input", "meta", "link"]
      ):
        # It's an open tag
        stack.append(el)
      elif isinstance(el, str):
        # It's a close tag name
        if stack and stack[-1].name == el:
          closed_tag = stack.pop()
          if stack:
            stack[-1].children.append(closed_tag)
          else:
            root_children.append(closed_tag)
      else:
        if stack:
          stack[-1].children.append(el)
        else:
          root_children.append(el)

    flat_elements.children = root_children

    # Update the document model name now that hierarchy is built
    model_name = "Model"
    from typing import Sequence

    def _find_model(nodes: Sequence[HtmlNode]) -> None:
      """Find the model name."""
      nonlocal model_name
      for n in nodes:
        if isinstance(n, TagNode):
          if n.name == "h3":
            full_text = "".join(c.content for c in n.children if isinstance(c, TextNode))
            if "Model:" in full_text:
              model_name = full_text.replace("Model:", "").strip()
          _find_model(n.children)

    _find_model(flat_elements.children)
    flat_elements.model_name = model_name

    return flat_elements

  def parse(self) -> cst.Module:
    """Execute implementation detail."""
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
        config_str = ""  # pragma: no cover

      target_api_class = f"dsl.{kind}"

      # Safely construct the RHS expression
      if not config_str:
        # Fallback for empty config
        rhs = cst.Call(func=self._create_dotted(target_api_class), args=[])  # pragma: no cover
      else:
        # Use robustness helper
        rhs = self._create_call(target_api_class, config_str=config_str)

      target = cst.Attribute(value=cst.Name("self"), attr=cst.Name(name))
      init_stmts.append(cst.SimpleStatementLine([cst.Assign(targets=[cst.AssignTarget(target)], value=rhs)]))

    if not init_stmts:
      # Pass if empty
      init_stmts.append(cst.SimpleStatementLine([cst.Pass()]))  # pragma: no cover

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
    """Execute implementation detail."""
    parts = name.split(".")
    node = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(node, cst.Name(p))  # type: ignore
    return node

  def _create_call(self, func_name: Any, config_str: Any = None) -> Any:
    """Execute implementation detail."""
    args = []
    if config_str:
      args = self._parse_args_str(config_str)
    return cst.Call(func=self._create_dotted(func_name), args=args)

  def _parse_args_str(self, s: str) -> List[cst.Arg]:
    """Parse key=val, key2=val2 string into CST Args."""
    if not s:
      return []  # pragma: no cover
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
        args.append(cst.Arg(self._safe_val(p.strip())))  # pragma: no cover
    return args

  def _safe_val(self, v: Any) -> Any:
    """Execute implementation detail."""
    try:
      return cst.parse_expression(v)
    except Exception:  # pragma: no cover
      return cst.SimpleString(f"'{v}'")  # pragma: no cover
