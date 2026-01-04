"""
HTML Parser.

Parses the HTML DSL structure using standard library `html.parser`.
Reconstructs a Python LibCST Module representing the logic described by the visual grid.

Logic:

1.  Scrapes `<div>` elements with specific classes (`box r`, `box b`) to identify
    memory allocations (Attributes) and computation steps (Operations).
2.  Extracts textual metadata from headers (`<span>`) and code blocks (`<code>`).
3.  Synthesizes a Python Class Definition using `libcst`, wrapping operations in
    a virtual `html_dsl` namespace (aliased as `dsl`).
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
    """Initialize parser state buffers."""
    super().__init__()
    # State Flags
    self.in_box = False
    self.current_class = ""
    self.in_header_txt = False
    self.in_code = False

    # Data Model
    self.model_name = "Model"
    self.attrs: List[Tuple[str, str, str]] = []  # (name, kind, config_str)
    self.ops: List[Tuple[str, str]] = []  # (op_name, args_str)

    # Temporary Text Buffers
    self._buf_header = ""
    self._buf_code = ""

  def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
    """
    Handles opening tags to set context flags.

    Args:
        tag: The tag name.
        attrs: List of attribute tuples.
    """
    attrs_dict = dict((k, v) for k, v in attrs if v is not None)
    cls = attrs_dict.get("class", "")

    if tag == "div" and "box" in cls:
      self.in_box = True
      self.current_class = cls
      # Reset buffers for new box
      self._buf_header = ""
      self._buf_code = ""

    if tag == "span" and "header-txt" in cls:
      self.in_header_txt = True

    if tag == "code":
      self.in_code = True

  def handle_endtag(self, tag: str) -> None:
    """
    Handles closing tags to process buffered content.

    Args:
        tag: The tag name.
    """
    if tag == "div" and self.in_box:
      self.in_box = False
      self._process_box()
      self.current_class = ""

    if tag == "span":
      self.in_header_txt = False

    if tag == "code":
      self.in_code = False

  def handle_data(self, data: str) -> None:
    """
    Accumulates text content based on active flags.

    Args:
        data: Text content within a tag.
    """
    if self.in_header_txt:
      self._buf_header += data
    if self.in_code:
      self._buf_code += data
    if "Model:" in data:
      # Basic title scraping. Assuming format "Model: Name"
      clean = data.replace("Model:", "").strip()
      if clean:
        self.model_name = clean

  def _process_box(self) -> None:
    """Interprets the buffered text of a closed box div."""
    txt = self._buf_header.strip()
    code = self._buf_code.strip()

    # Split classes to check for exact tokens (e.g. "box" vs "b" vs "g")
    # This prevents "box g" (contains 'b' char) matching Blue box logic
    classes = self.current_class.strip().split()

    # Red Box (Memory/Attributes)
    # Class: "box r"
    # Format: Header="name: Kind", Code="arg1=val, ..."
    if "r" in classes:
      if ":" in txt:
        name, kind = txt.split(":", 1)
        self.attrs.append((name.strip(), kind.strip(), code))
      else:
        # Fallback for malformed headers
        self.attrs.append((txt, "Unknown", code))

    # Blue Box (Operations)
    # Class: "box b"
    # Format: Header="OpName" or "Call (attr)", Code="args: val, ..."
    elif "b" in classes:
      op = txt
      # Strip "args:" prefix from code block if present
      args = code.replace("args:", "").strip()
      self.ops.append((op, args))

    # Green "Data" boxes ("box g") are intentionally ignored


class HtmlParser:
  """
  Facade for parsing HTML strings into LibCST modules.
  """

  def __init__(self, source: str) -> None:
    """
    Initializes the parser.

    Args:
        source (str): The raw HTML string containing the grid DSL.
    """
    self.source = source

  def parse(self) -> cst.Module:
    """
    Parses the HTML and synthesizes a Python AST.

    Returns:
        cst.Module: A Python module definition of the neural network class.
    """
    p = GridParser()
    p.feed(self.source)

    # 1. Imports
    # Inject standard import alias for the DSL
    import_stmt = cst.parse_statement("import html_dsl as dsl")
    body_stmts: List[cst.CSTNode] = [import_stmt]

    # 2. Build __init__
    init_stmts = []
    if p.attrs:
      # If attributes exist, call super init
      init_stmts.append(cst.parse_statement("super().__init__()"))

    for name, kind, cfg in p.attrs:
      # Synthesize assignment: self.name = dsl.Kind(cfg)
      # We assume 'cfg' is valid python argument syntax (e.g. "k=3, s=1")
      if not cfg:
        call_code = f"self.{name} = dsl.{kind}()"
      else:
        call_code = f"self.{name} = dsl.{kind}({cfg})"

      try:
        stmt = cst.parse_statement(call_code)
        init_stmts.append(stmt)
      except cst.ParserSyntaxError:
        # Fallback for invalid config strings
        pass

    if not init_stmts:
      init_stmts.append(cst.parse_statement("pass"))

    # 3. Build forward
    fwd_stmts = []
    # 'x' is the standard input argument in generated code
    last_var = "x"

    for op_name, args in p.ops:
      # Check for Stateful Call signature: "Call (attr_name)"
      if op_name.startswith("Call (") and op_name.endswith(")"):
        attr = op_name[6:-1]  # Strip "Call (" and ")"

        # Input to call: Usually the previous variable
        # For stateful calls, args in HTML might be "x" (default placeholder)
        # We chain the logic: out = self.attr(last_var)
        call_var = last_var

        # New output variable name
        out_var = f"{attr}_out"

        line = f"{out_var} = self.{attr}({call_var})"
        fwd_stmts.append(cst.parse_statement(line))
        last_var = out_var

      else:
        # Functional Op: Flatten, Relu
        # Construct: out = dsl.OpName(input, other_args)
        # We need to prepend the input variable to the args list logic
        # HTML Args string: "start_dim=1" or just "x"

        # If args string is empty/default, use last_var
        # If args string has content, we must determine if input var is implicit
        # In this DSL, Blue boxes usually list auxiliary args.
        # Implicit chaining convention:

        # e.g. "args: s1, start=1" -> means input is s1.
        # If args is empty or implicit, assume last_var is input.

        # NOTE: Parser relies on simple AST injection. Robust argument merging
        # is complex without knowing op signature.
        # Strategy: If "args" contains variable name, use it. Else inject last_var.

        final_args = args
        if not final_args or final_args == "x":
          final_args = last_var
        else:
          # If args provided (e.g. "dim=1"), prepend input
          # Check if first arg looks like variable
          if "=" in final_args.split(",")[0]:
            final_args = f"{last_var}, {final_args}"

        out_var = f"op_{len(fwd_stmts)}"

        # Function name capitalization from HTML might need adjustment?
        # "Relu" -> dsl.Relu. "Flatten" -> dsl.Flatten.
        target_api = f"dsl.{op_name}"

        line = f"{out_var} = {target_api}({final_args})"
        try:
          fwd_stmts.append(cst.parse_statement(line))
          last_var = out_var
        except cst.ParserSyntaxError:
          pass

    # Return statement
    fwd_stmts.append(cst.parse_statement(f"return {last_var}"))

    # 4. Construct Class
    class_def = cst.ClassDef(
      name=cst.Name(p.model_name),
      bases=[cst.Arg(cst.parse_expression("dsl.Module"))],
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

    body_stmts.append(class_def)
    return cst.Module(body=body_stmts)
