"""WebAssembly Text Format (WAT) Concrete Syntax Tree nodes.

Provides a structural representation for parsing and emitting .wat files.
"""

import re
import typing

import pydantic

WasmOpcode = typing.NewType("WasmOpcode", str)
WasmRegister = typing.NewType("WasmRegister", str)
WasmArgument = typing.NewType("WasmArgument", str)


class WatNode(pydantic.BaseModel):
  """Base node for all WAT components."""

  model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  def to_text(self, indent: int = 0) -> str:
    """Render the component to WAT text.

    Args:
        indent: Indentation level.

    Returns:
        The textual representation of the node.
    """
    raise NotImplementedError()


class WatParam(WatNode):
  """Represent a parameter in a WAT function.

  Attributes:
      name: The register name of the parameter.
      type_id: The type identifier of the parameter.
  """

  name: WasmRegister
  type_id: str

  def __init__(self, name: WasmRegister, type_id: str) -> None:
    """Initialize WatParam.

    Args:
        name: The register name of the parameter.
        type_id: The type identifier of the parameter.
    """
    pydantic.BaseModel.__init__(self, name=name, type_id=type_id)

  def to_text(self, indent: int = 0) -> str:
    """Render the param.

    Args:
        indent: Indentation level.

    Returns:
        The rendered parameter string.
    """
    return f"(param ${self.name} {self.type_id})"


class WatResult(WatNode):
  """Represent a return result in a WAT function.

  Attributes:
      type_id: The type identifier of the return result.
  """

  type_id: str

  def __init__(self, type_id: str) -> None:
    """Initialize WatResult.

    Args:
        type_id: The type identifier of the return result.
    """
    pydantic.BaseModel.__init__(self, type_id=type_id)

  def to_text(self, indent: int = 0) -> str:
    """Render the result.

    Args:
        indent: Indentation level.

    Returns:
        The rendered result string.
    """
    return f"(result {self.type_id})"


class WatLocal(WatNode):
  """Represent a local variable definition in a WAT function.

  Attributes:
      name: The register name of the local variable.
      type_id: The type identifier of the local variable.
  """

  name: WasmRegister
  type_id: str

  def __init__(self, name: WasmRegister, type_id: str) -> None:
    """Initialize WatLocal.

    Args:
        name: The register name of the local variable.
        type_id: The type identifier of the local variable.
    """
    pydantic.BaseModel.__init__(self, name=name, type_id=type_id)

  def to_text(self, indent: int = 0) -> str:
    """Render the local.

    Args:
        indent: Indentation level.

    Returns:
        The rendered local variable string.
    """
    return f"(local ${self.name} {self.type_id})"


class WatInstr(WatNode):
  """Represent a single WebAssembly instruction.

  Attributes:
      opcode: The instruction operation code.
      args: List of WebAssembly arguments for the instruction.
  """

  opcode: WasmOpcode
  args: typing.List[WasmArgument] = pydantic.Field(default_factory=list)

  def __init__(self, opcode: WasmOpcode, args: typing.Optional[typing.List[WasmArgument]] = None) -> None:
    """Initialize WatInstr.

    Args:
        opcode: The instruction operation code.
        args: List of WebAssembly arguments for the instruction.
    """
    pydantic.BaseModel.__init__(self, opcode=opcode, args=args if args is not None else [])

  def to_text(self, indent: int = 0) -> str:
    """Render the instruction.

    Args:
        indent: Indentation level.

    Returns:
        The rendered instruction string.
    """
    sp = "  " * indent
    args_str = " ".join(self.args)
    if args_str:
      return f"{sp}{self.opcode} {args_str}"
    return f"{sp}{self.opcode}"


class WatFunc(WatNode):
  """Represent a WebAssembly function.

  Attributes:
      name: Name of the function.
      params: List of function parameters.
      results: List of return results.
      locals: List of local variables.
      body: List of instructions inside the function.
      export: Flag indicating if the function is exported.
  """

  name: str
  params: typing.List[WatParam] = pydantic.Field(default_factory=list)
  results: typing.List[WatResult] = pydantic.Field(default_factory=list)
  locals: typing.List[WatLocal] = pydantic.Field(default_factory=list)
  body: typing.List[WatInstr] = pydantic.Field(default_factory=list)
  export: bool = False

  def __init__(
    self,
    name: str,
    params: typing.Optional[typing.List[WatParam]] = None,
    results: typing.Optional[typing.List[WatResult]] = None,
    locals: typing.Optional[typing.List[WatLocal]] = None,
    body: typing.Optional[typing.List[WatInstr]] = None,
    export: bool = False,
  ) -> None:
    """Initialize WatFunc.

    Args:
        name: Name of the function.
        params: List of function parameters.
        results: List of return results.
        locals: List of local variables.
        body: List of instructions inside the function.
        export: Flag indicating if the function is exported.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      params=params if params is not None else [],
      results=results if results is not None else [],
      locals=locals if locals is not None else [],
      body=body if body is not None else [],
      export=export,
    )

  def to_text(self, indent: int = 0) -> str:
    """Render the function.

    Args:
        indent: Indentation level.

    Returns:
        The rendered function block string.
    """
    sp = "  " * indent
    parts = [f"{sp}(func ${self.name}"]

    if self.export:
      parts.append(f' (export "{self.name}")')

    if self.params:
      parts.append(" " + " ".join(p.to_text() for p in self.params))

    if self.results:
      parts.append(" " + " ".join(r.to_text() for r in self.results))

    parts.append("\n")

    inner_indent = indent + 1
    for loc in self.locals:
      parts.append("  " * inner_indent + loc.to_text() + "\n")

    for instr in self.body:
      parts.append(instr.to_text(inner_indent) + "\n")

    parts.append(f"{sp})")
    return "".join(parts)


class WatModule(WatNode):
  """Represent a WebAssembly module.

  Attributes:
      functions: List of functions in the module.
  """

  functions: typing.List[WatFunc] = pydantic.Field(default_factory=list)

  def __init__(self, functions: typing.Optional[typing.List[WatFunc]] = None) -> None:
    """Initialize WatModule.

    Args:
        functions: List of functions in the module.
    """
    pydantic.BaseModel.__init__(self, functions=functions if functions is not None else [])

  def to_text(self, indent: int = 0) -> str:
    """Render the module.

    Args:
        indent: Indentation level.

    Returns:
        The rendered module block string.
    """
    parts = ["(module\n"]
    for func in self.functions:
      parts.append(func.to_text(1) + "\n")
    parts.append(")\n")
    return "".join(parts)


class WatParser:
  """Parse WAT strings into WatModule CST."""

  def __init__(self, text: str) -> None:
    """Initialize parser with source text.

    Args:
        text: The source WebAssembly text representation.
    """
    self.text = text
    self.pos = 0
    self.tokens = self._tokenize()

  def _tokenize(self) -> typing.List[str]:
    """Tokenize the WAT text into S-expression components.

    Returns:
        A list of parsed token strings.
    """
    tokens = []
    pos = 0
    while pos < len(self.text):
      # skip whitespace
      match = re.match(r"\s+", self.text[pos:])
      if match:
        pos += match.end()
        continue

      # skip comments
      match = re.match(r";;.*", self.text[pos:])
      if match:
        pos += match.end()
        continue

      # parens
      if self.text[pos] in "()":
        tokens.append(self.text[pos])
        pos += 1
        continue

      # strings
      if self.text[pos] == '"':
        match = re.match(r'"([^"\\]*(\\.[^"\\]*)*)"', self.text[pos:])
        if match:
          tokens.append(match.group(0))
          pos += match.end()
          continue

      # identifiers, numbers, keywords
      match = re.match(r'[^\s()";]+', self.text[pos:])
      if match:
        tokens.append(match.group(0))
        pos += match.end()
        continue

      # fallback
      tokens.append(self.text[pos])
      pos += 1
    return tokens

  def _peek(self) -> str:
    """Peek the next token.

    Returns:
        The next token string, or empty string if at EOF.
    """
    if self.pos < len(self.tokens):
      return self.tokens[self.pos]
    return ""

  def _consume(self, expected: str = "") -> str:
    """Consume the next token, optionally verifying it.

    Args:
        expected: Optional string token to compare against.

    Returns:
        The consumed token string.

    Raises:
        ValueError: If EOF is hit unexpectedly or expected token is mismatched.
    """
    if self.pos >= len(self.tokens):
      raise ValueError("Unexpected EOF")
    tok = self.tokens[self.pos]
    if expected and tok != expected:
      raise ValueError(f"Expected {expected}, got {tok}")
    self.pos += 1
    return tok

  def _skip_to_matching_paren(self) -> None:
    """Skip an S-expression block."""
    self._consume("(")
    depth = 1
    while depth > 0 and self.pos < len(self.tokens):
      tok = self._consume()
      if tok == "(":
        depth += 1
      elif tok == ")":
        depth -= 1

  def parse(self) -> WatModule:
    """Parse the WAT tokens into a WatModule.

    Returns:
        The constructed WatModule representation.
    """
    self._consume("(")
    self._consume("module")
    module = WatModule()
    while self._peek() == "(":
      saved_pos = self.pos
      self._consume("(")
      kw = self._consume()
      if kw == "func":
        self.pos = saved_pos
        module.functions.append(self._parse_func())
      else:
        self.pos = saved_pos
        self._skip_to_matching_paren()
    self._consume(")")
    return module

  def _parse_func(self) -> WatFunc:
    """Parse a function block.

    Returns:
        The parsed WatFunc block representation.
    """
    self._consume("(")
    self._consume("func")
    func_name = ""
    name_tok = self._peek()
    if name_tok.startswith("$"):
      func_name = self._consume()[1:]

    export = False
    params = []
    results = []
    locals_ = []
    body = []

    # parse structural blocks
    while self._peek() == "(":
      saved_pos = self.pos
      self._consume("(")
      kw = self._consume()
      if kw == "export":
        export = True
        self._skip_to_matching_paren_from(saved_pos)
      elif kw == "param":
        p_name = self._consume()
        if p_name.startswith("$"):
          p_name = p_name[1:]
          type_id = self._consume()
        else:
          type_id = p_name
          p_name = ""
        params.append(WatParam(WasmRegister(p_name), type_id))
        self._consume(")")
      elif kw == "result":
        type_id = self._consume()
        results.append(WatResult(type_id))
        self._consume(")")
      elif kw == "local":
        l_name = self._consume()
        if l_name.startswith("$"):
          l_name = l_name[1:]
          type_id = self._consume()
        else:
          type_id = l_name
          l_name = ""
        locals_.append(WatLocal(WasmRegister(l_name), type_id))
        self._consume(")")
      else:
        self.pos = saved_pos  # Not a recognized header element, maybe an instruction?
        break

    # parse instructions
    while self._peek() != ")":
      tok = self._consume()
      # if instruction uses s-expression (e.g. `(local.get $x)`)
      if tok == "(":
        instr_op = self._consume()
        args = []
        while self._peek() != ")":
          arg = self._consume()
          args.append(WasmArgument(arg))
        self._consume(")")
        body.append(WatInstr(WasmOpcode(instr_op), args))
      else:
        # flat instruction
        instr_op = tok
        args = []
        # consume args until next instruction or end of func
        while (
          self._peek() not in (")", "(")
          and not re.match(r"^[a-z0-9]+\.[a-z0-9_]+$", self._peek())
          and self._peek() not in ("nop", "drop", "return", "unreachable", "end", "else")
        ):
          args.append(WasmArgument(self._consume()))
        body.append(WatInstr(WasmOpcode(instr_op), args))

    self._consume(")")
    return WatFunc(
      name=func_name,
      export=export,
      params=params,
      results=results,
      locals=locals_,
      body=body,
    )

  def _skip_to_matching_paren_from(self, saved_pos: int) -> None:
    """Skip to matching paren.

    Args:
        saved_pos: The saved position in token parsing context.
    """
    self.pos = saved_pos
    self._skip_to_matching_paren()
