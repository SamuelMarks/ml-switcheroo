"""C++ Parser.

A pure Python Lark-based parser for a subset of C++
used in PyTorch extensions, producing CppNode CSTs.
"""

from typing import List, Any

from lark import Lark, Transformer, v_args

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  CppModule,
  IncludeDirective,
  MacroDefinition,
  FunctionDefinition,
  FunctionArgument,
  TypeIdentifier,
  VariableDeclaration,
  ReturnStatement,
  Identifier,
  BinaryExpression,
  MethodCall,
  RawStatement,
  PyBindModule,
  PyBindDef,
  Expression,
  CppNode,
)


GRAMMAR = r"""
    ?start: module | statement
    module: (include | macro | pybind | function | statement)*

    include: "#include" WS* "<" IDENTIFIER_PATH ">" -> include_system
           | "#include" WS* "\"" IDENTIFIER_PATH "\"" -> include_local

    macro: "#define" WS+ IDENTIFIER WS+ STRING -> macro_define
         | "#define" WS+ IDENTIFIER WS+ NUMBER -> macro_define
         | "#define" WS+ IDENTIFIER -> macro_empty

    pybind: "PYBIND11_MODULE" WS* "(" WS* IDENTIFIER WS* "," WS* IDENTIFIER WS* ")" WS* "{" pybind_def* "}"
    pybind_def: IDENTIFIER WS* "." WS* "def" WS* "(" WS* STRING WS* "," WS* "&" WS* IDENTIFIER WS* "," WS* STRING WS* ")" WS* ";"

    function: type_id WS* IDENTIFIER WS* "(" [func_args] ")" WS* "{" statement* "}"

    func_args: func_arg ("," WS* func_arg)*
    func_arg: type_id WS* IDENTIFIER

    ?statement: "return" WS* ";" -> return_empty
              | "return" WS+ expression ";" -> return_expr
              | type_id WS* IDENTIFIER WS* ";" -> var_decl
              | type_id WS* IDENTIFIER WS* "=" WS* expression ";" -> var_decl_init
              | raw_statement

    raw_statement: /[^;{}]+/ ";"

    ?expression: method_call | binary_expr | identifier | number_lit | string_lit

    method_call: identifier "(" [expression ("," WS* expression)*] ")"
    binary_expr: expression WS* PUNCT WS* expression
    identifier: IDENTIFIER
    number_lit: NUMBER
    string_lit: STRING
    type_id: IDENTIFIER

    IDENTIFIER_PATH: /[a-zA-Z0-9_\/\.]+/
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_:]*/
    NUMBER: /-?\d+(?:\.\d+)?/
    STRING: /"(?:[^"\\]|\\.)*"/
    PUNCT: /[+\-*\/&]/
    WS: /[ \t\f\r\n]+/

    %ignore WS
    %ignore /\/\/[^\n]*/
    %ignore /\/\*.*?\*\//s
"""


class CppTransformer(Transformer[Any, Any]):
  """Transforms parsed AST nodes into CppNode classes."""

  @v_args(inline=False)
  def module(self, children: List[Any]) -> CppModule:
    """Transform the top-level module rule.

    Args:
          children: The parsed child nodes of the module, representing statements,
            functions, macros, or imports.

    Returns:
        CppModule: A CppModule instance holding the grouped includes and other body nodes.
    """
    includes = [c for c in children if isinstance(c, IncludeDirective)]
    body = [c for c in children if not isinstance(c, IncludeDirective) and c is not None]
    return CppModule(includes=includes, body=body)

  @v_args(inline=False)
  def include_system(self, children: List[Any]) -> IncludeDirective:
    """Transform a system include.

    Args:
          children: The parsed child nodes, which should contain the system header path identifier.

    Returns:
        IncludeDirective: An IncludeDirective representing a system include (e.g., <vector>).

    Raises:
        AssertionError: If child token is missing.
    """
    for c in children:
      if getattr(c, "type", None) == "IDENTIFIER_PATH":
        return IncludeDirective(path=c.value, system=True)
    raise AssertionError("No IDENTIFIER_PATH found")

  @v_args(inline=False)
  def include_local(self, children: List[Any]) -> IncludeDirective:
    """Transform a local include.

    Args:
          children: The parsed child nodes, which should contain the local header path identifier.

    Returns:
        IncludeDirective: An IncludeDirective representing a local include (e.g., "my_header.h").

    Raises:
        AssertionError: If child token is missing.
    """
    for c in children:
      if getattr(c, "type", None) == "IDENTIFIER_PATH":
        return IncludeDirective(path=c.value, system=False)
    raise AssertionError("No IDENTIFIER_PATH found")

  @v_args(inline=False)
  def macro_define(self, children: List[Any]) -> MacroDefinition:
    """Transform a macro definition with a value.

    Args:
          children: The parsed child nodes containing the macro identifier and its value (string/number).

    Returns:
        MacroDefinition: A MacroDefinition representing the macro with its name and defined value.
    """
    name, val = "", ""
    for c in children:
      if getattr(c, "type", None) == "IDENTIFIER":
        name = c.value
      elif getattr(c, "type", None) in ("STRING", "NUMBER"):
        val = c.value
    return MacroDefinition(name=name, value=val)

  @v_args(inline=False)
  def macro_empty(self, children: List[Any]) -> MacroDefinition:
    """Transform an empty macro definition.

    Args:
          children: The parsed child nodes containing the macro identifier.

    Returns:
        MacroDefinition: A MacroDefinition representing the empty macro with its name and an empty value string.
    """
    name = ""
    for c in children:
      if getattr(c, "type", None) == "IDENTIFIER":
        name = c.value
    return MacroDefinition(name=name, value="")

  @v_args(inline=False)
  def function(self, children: List[Any]) -> FunctionDefinition:
    """Transform a function definition.

    Args:
          children: The parsed child nodes including the return type, function name,
            optional arguments, and the statements in the function body.

    Returns:
        FunctionDefinition: A FunctionDefinition representing the full C++ function definition.
    """
    ret_type = children[0]
    name = ""
    for c in children[1:]:
      if getattr(c, "type", None) == "IDENTIFIER":
        name = c.value
        break

    args = []
    body = []
    for c in children:
      if isinstance(c, list):
        args = c
      elif isinstance(c, CppNode) and not isinstance(c, TypeIdentifier):
        body.append(c)
    return FunctionDefinition(return_type=ret_type, name=name, arguments=args, body=body)

  @v_args(inline=False)
  def func_args(self, children: List[Any]) -> List[FunctionArgument]:
    """Transform function arguments list.

    Args:
          children: The list of parsed individual function arguments.

    Returns:
        List[FunctionArgument]: A list of FunctionArgument nodes.
    """
    return [c for c in children if isinstance(c, FunctionArgument)]

  @v_args(inline=False)
  def func_arg(self, children: List[Any]) -> FunctionArgument:
    """Transform a single function argument.

    Args:
          children: The parsed child nodes containing the argument type identifier and name.

    Returns:
        FunctionArgument: A FunctionArgument representing the argument.
    """
    name = ""
    for c in children[1:]:
      if getattr(c, "type", None) == "IDENTIFIER":
        name = c.value
        break
    return FunctionArgument(type_id=children[0], name=name)

  @v_args(inline=False)
  def type_id(self, children: List[Any]) -> TypeIdentifier:
    """Transform a type identifier.

    Args:
          children: A list containing the parsed type identifier token.

    Returns:
        TypeIdentifier: A TypeIdentifier AST node containing the type name.
    """
    return TypeIdentifier(name=children[0].value)

  @v_args(inline=False)
  def pybind(self, children: List[Any]) -> PyBindModule:
    """Transform a PYBIND11_MODULE.

    Args:
          children: The parsed child nodes of the pybind module, containing the module
            name, variable name, and optional pybind definitions.

    Returns:
        PyBindModule: A PyBindModule node representing the module export.
    """
    name = ""
    module_var = ""
    for c in children:
      if getattr(c, "type", None) == "IDENTIFIER":
        if not name:
          name = c.value
        else:
          module_var = c.value
          break
    defs = [c for c in children if isinstance(c, PyBindDef)]
    return PyBindModule(name=name, module_var=module_var, defs=defs)

  @v_args(inline=False)
  def pybind_def(self, children: List[Any]) -> PyBindDef:
    """Transform a m.def() call.

    Args:
          children: The parsed child nodes of the pybind method definition, including
            the exported name, the function reference, and the docstring.

    Returns:
        PyBindDef: A PyBindDef node representing the registered pybind method.
    """
    strings = [c.value.strip('"') for c in children if getattr(c, "type", None) == "STRING"]
    idents = [c.value for c in children if getattr(c, "type", None) == "IDENTIFIER"]
    # idents[0] is `m`
    # idents[1] is `f_ref` (since "def" and "&" are string terminals and omitted by Lark)
    return PyBindDef(name=strings[0], function_ref=idents[1], docstring=strings[1])

  @v_args(inline=False)
  def return_empty(self, children: List[Any]) -> ReturnStatement:
    """Transform an empty return statement.

    Args:
          children: The parsed children (typically empty or containing semicolon/return tokens).

    Returns:
        ReturnStatement: A ReturnStatement representing a void return statement.
    """
    return ReturnStatement()

  @v_args(inline=False)
  def return_expr(self, children: List[Any]) -> ReturnStatement:
    """Transform a return statement with an expression.

    Args:
          children: The parsed children containing the expression to return.

    Returns:
        ReturnStatement: A ReturnStatement containing the returned expression.
    """
    expr = next(c for c in children if isinstance(c, Expression))
    return ReturnStatement(value=expr)

  @v_args(inline=False)
  def var_decl(self, children: List[Any]) -> VariableDeclaration:
    """Transform a variable declaration.

    Args:
          children: The parsed child nodes including the variable's type and its name identifier.

    Returns:
        VariableDeclaration: A VariableDeclaration node without an initializer.
    """
    name = ""
    for c in children[1:]:
      if getattr(c, "type", None) == "IDENTIFIER":
        name = c.value
    return VariableDeclaration(type_id=children[0], name=name)

  @v_args(inline=False)
  def var_decl_init(self, children: List[Any]) -> VariableDeclaration:
    """Transform a variable declaration with initialization.

    Args:
          children: The parsed child nodes including the variable's type, its name identifier,
            and the initial value expression.

    Returns:
        VariableDeclaration: A VariableDeclaration node with its associated initializer expression.
    """
    name = ""
    for c in children[1:]:
      if getattr(c, "type", None) == "IDENTIFIER":
        name = c.value
        break
    expr = None
    for c in children:
      if isinstance(c, Expression):
        expr = c
        break
    return VariableDeclaration(type_id=children[0], name=name, initializer=expr)

  @v_args(inline=False)
  def raw_statement(self, children: List[Any]) -> RawStatement:
    """Transform a raw statement.

    Args:
          children: The parsed child token representing the raw unparsed C++ statement.

    Returns:
        RawStatement: A RawStatement containing the raw string code.
    """
    return RawStatement(code=children[0].value.strip())

  @v_args(inline=False)
  def identifier(self, children: List[Any]) -> Identifier:
    """Transform an identifier.

    Args:
          children: The parsed token containing the identifier name.

    Returns:
        Identifier: An Identifier AST node wrapping the name.
    """
    return Identifier(name=children[0].value)

  @v_args(inline=False)
  def number_lit(self, children: List[Any]) -> Identifier:
    """Transform a number literal into an identifier.

    Args:
          children: The parsed token containing the numeric literal.

    Returns:
        Identifier: An Identifier AST node wrapping the number literal as its name.
    """
    return Identifier(name=children[0].value)

  @v_args(inline=False)
  def string_lit(self, children: List[Any]) -> Identifier:
    """Transform a string literal into an identifier.

    Args:
          children: The parsed token containing the string literal.

    Returns:
        Identifier: An Identifier AST node wrapping the string literal as its name.
    """
    return Identifier(name=children[0].value)

  @v_args(inline=False)
  def method_call(self, children: List[Any]) -> MethodCall:
    """Transform a method call.

    Args:
          children: The parsed child nodes representing the called function/method name,
            followed by its arguments.

    Returns:
        MethodCall: A MethodCall AST node wrapping the name and arguments.
    """
    name = children[0].name
    args = [c for c in children[1:] if isinstance(c, Expression)]
    return MethodCall(name=name, arguments=args)

  @v_args(inline=False)
  def binary_expr(self, children: List[Any]) -> BinaryExpression:
    """Transform a binary expression.

    Args:
          children: The parsed child nodes containing the left operand, operator token, and right operand.

    Returns:
        BinaryExpression: A BinaryExpression AST node.
    """
    op = ""
    for c in children:
      if getattr(c, "type", None) == "PUNCT":
        op = c.value
        break
    exprs = [c for c in children if isinstance(c, Expression)]
    return BinaryExpression(left=exprs[0], operator=op, right=exprs[1])


class CppParser:
  """A parser for C++ source code using Lark."""

  def __init__(self, text: str):
    """Initialize the parser with the given source code.

    Args:
        text: The raw C++ source code string to be parsed.
    """
    self.text = text
    self.parser = Lark(GRAMMAR, parser="earley")
    self.transformer = CppTransformer()

  def parse(self) -> CppModule:
    """Parse the source code and return a CppModule.

    Returns:
        CppModule: A parsed CppModule representing the abstract syntax tree of the C++ code.

    Raises:
        ValueError: If parsing fails or an unexpected token is encountered.
    """
    if not self.text.strip():
      return CppModule()

    try:
      tree = self.parser.parse(self.text)
      return self.transformer.transform(tree)  # type: ignore
    except Exception as e:
      raise ValueError(f"Unexpected token: {e}")
