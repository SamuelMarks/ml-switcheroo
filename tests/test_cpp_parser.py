"""Test module."""

import pytest

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  BinaryExpression,
  CppModule,
  FunctionDefinition,
  Identifier,
  MacroDefinition,
  MethodCall,
  PyBindModule,
  RawStatement,
  ReturnStatement,
  VariableDeclaration,
)
from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser


def test_parser_empty() -> None:
  """Docstring."""
  parser: CppParser = CppParser("   \n ")
  node: CppModule = parser.parse()
  assert isinstance(node, CppModule)
  assert len(node.includes) == 0
  assert len(node.body) == 0


def test_parser_include_system() -> None:
  """Docstring."""
  parser: CppParser = CppParser("#include <iostream>")
  node: CppModule = parser.parse()
  assert isinstance(node, CppModule)
  assert len(node.includes) == 1
  assert node.includes[0].path == "iostream"
  assert node.includes[0].system is True


def test_parser_include_local() -> None:
  """Docstring."""
  parser: CppParser = CppParser('#include "my_header.h"')
  node: CppModule = parser.parse()
  assert isinstance(node, CppModule)
  assert len(node.includes) == 1
  assert node.includes[0].path == "my_header.h"
  assert node.includes[0].system is False


def test_parser_macro_define() -> None:
  """Docstring."""
  parser: CppParser = CppParser("#define FOO 1")
  node: CppModule = parser.parse()
  assert len(node.body) == 1
  assert isinstance(node.body[0], MacroDefinition)
  assert node.body[0].name == "FOO"
  assert node.body[0].value == "1"

  parser_str: CppParser = CppParser('#define BAR "test"')
  node_str: CppModule = parser_str.parse()
  macro = node_str.body[0]
  assert isinstance(macro, MacroDefinition)
  assert macro.name == "BAR"
  assert macro.value == '"test"'


def test_parser_macro_empty() -> None:
  """Docstring."""
  parser: CppParser = CppParser("#define BAZ")
  node: CppModule = parser.parse()
  assert len(node.body) == 1
  assert isinstance(node.body[0], MacroDefinition)
  assert node.body[0].name == "BAZ"
  assert node.body[0].value == ""


def test_parser_function() -> None:
  """Docstring."""
  code: str = """
    int main() {
        return 0;
    }
    """
  parser: CppParser = CppParser(code)
  node: CppModule = parser.parse()
  assert len(node.body) == 1
  func = node.body[0]
  assert isinstance(func, FunctionDefinition)
  assert func.return_type.name == "int"
  assert func.name == "main"
  assert len(func.arguments) == 0
  assert len(func.body) == 1
  assert isinstance(func.body[0], ReturnStatement)
  # body[0].value could be None but in this case it's Identifier
  val = func.body[0].value
  assert isinstance(val, Identifier)
  assert val.name == "0"


def test_parser_function_args() -> None:
  """Docstring."""
  code: str = """
    void foo(int a, float b) {
    }
    """
  parser: CppParser = CppParser(code)
  node: CppModule = parser.parse()
  func = node.body[0]
  assert isinstance(func, FunctionDefinition)
  assert len(func.arguments) == 2
  assert func.arguments[0].type_id.name == "int"
  assert func.arguments[0].name == "a"
  assert func.arguments[1].type_id.name == "float"
  assert func.arguments[1].name == "b"


def test_parser_return_empty() -> None:
  """Docstring."""
  code: str = """
    void foo() {
        return;
    }
    """
  parser: CppParser = CppParser(code)
  node: CppModule = parser.parse()
  func = node.body[0]
  assert isinstance(func, FunctionDefinition)
  assert len(func.body) == 1
  assert isinstance(func.body[0], ReturnStatement)
  assert func.body[0].value is None


def test_parser_var_decl() -> None:
  """Docstring."""
  code: str = """
    int a;
    float b = 1.0;
    """
  parser: CppParser = CppParser(code)
  node: CppModule = parser.parse()
  assert len(node.body) == 2
  assert isinstance(node.body[0], VariableDeclaration)
  assert node.body[0].name == "a"
  assert node.body[0].initializer is None

  assert isinstance(node.body[1], VariableDeclaration)
  assert node.body[1].name == "b"
  assert isinstance(node.body[1].initializer, Identifier)
  assert node.body[1].initializer.name == "1.0"


def test_parser_raw_statement() -> None:
  """Docstring."""
  code: str = """
    std::cout << "hello";
    """
  parser: CppParser = CppParser(code)
  node: CppModule = parser.parse()
  assert len(node.body) == 1
  assert isinstance(node.body[0], RawStatement)
  assert node.body[0].code == 'std::cout << "hello"'


def test_parser_expressions() -> None:
  """Docstring."""
  code: str = """
    int main() {
        foo(1, a);
        int x = a + b;
    }
    """
  parser: CppParser = CppParser(code)
  node: CppModule = parser.parse()
  func = node.body[0]
  assert isinstance(func, FunctionDefinition)

  assert isinstance(func.body[0], RawStatement)  # 'foo(1, a)' is raw? Wait, is it?
  # Ah, method_call can be an expression, but if it stands alone it might not parse as expression unless in raw_statement or return...
  # actually, expression statement is not in grammar except as raw_statement.

  assert isinstance(func.body[1], VariableDeclaration)
  assert isinstance(func.body[1].initializer, BinaryExpression)


def test_parser_pybind() -> None:
  """Docstring."""
  code: str = """
    PYBIND11_MODULE(my_mod, m) {
        m.def("foo", &foo_impl, "doc");
    }
    """
  parser: CppParser = CppParser(code)
  node: CppModule = parser.parse()
  assert len(node.body) == 1
  mod = node.body[0]
  assert isinstance(mod, PyBindModule)
  assert mod.name == "my_mod"
  assert mod.module_var == "m"
  assert len(mod.defs) == 1
  assert mod.defs[0].name == "foo"
  assert mod.defs[0].function_ref == "foo_impl"
  assert mod.defs[0].docstring == "doc"


def test_parser_invalid() -> None:
  """Docstring."""
  parser: CppParser = CppParser("invalid syntax")
  with pytest.raises(ValueError):
    parser.parse()


def test_system_include_no_path() -> None:
  """Docstring."""
  from lark import Token

  from ml_switcheroo.core.compiler.backends.cpp.parser import CppTransformer

  transformer: CppTransformer = CppTransformer()
  import pytest

  with pytest.raises(AssertionError, match="No IDENTIFIER_PATH found"):
    transformer.include_system([Token("OTHER", "test")])


def test_local_include_no_path() -> None:
  """Docstring."""
  from lark import Token

  from ml_switcheroo.core.compiler.backends.cpp.parser import CppTransformer

  transformer: CppTransformer = CppTransformer()
  import pytest

  with pytest.raises(AssertionError, match="No IDENTIFIER_PATH found"):
    transformer.include_local([Token("OTHER", "test")])


def test_method_call() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.cpp.parser import CppTransformer

  transformer: CppTransformer = CppTransformer()
  res: MethodCall = transformer.method_call([Identifier(name="func_name"), Identifier(name="a")])
  assert res.name == "func_name"
  assert len(res.arguments) == 1
  assert res.arguments[0].name == "a"


def test_identifier_string_literal() -> None:
  """Docstring."""
  from lark import Token

  from ml_switcheroo.core.compiler.backends.cpp.parser import CppTransformer

  transformer: CppTransformer = CppTransformer()
  res: Identifier = transformer.string_lit([Token("STRING", '"hello"')])
  assert res.name == '"hello"'
