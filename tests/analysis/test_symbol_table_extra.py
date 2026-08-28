"""Test suite for symbol table analysis extra coverage."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer


def analyze(code: str) -> SymbolTableAnalyzer:
  """Analyze code for symbol table."""
  tree: cst.Module = cst.parse_module(code)
  sm: MagicMock = MagicMock()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(sm)
  tree.visit(analyzer)
  return analyzer


def test_missing_symbol_table_coverage() -> None:
  """Test missing symbol table coverage."""
  # Test try/except blocks
  code: str = """
try:
    x = 1
except Exception as e:
    x = 2
finally:
    y = 3
    """
  analyze(code)

  # Test boolean ops
  code_bool: str = """
x = True and False or True
    """
  analyze(code_bool)

  # Test unary ops
  code_unary: str = """
x = not True
y = -1
    """
  analyze(code_unary)

  # Test with/async with
  code_with: str = """
with open('file.txt') as f:
    x = 1
    """
  analyze(code_with)


def test_global_scope_access() -> None:
  """Test global scope access."""
  code: str = """
global_var = 1
def func():
    return global_var
    """
  analyze(code)


def test_class_def_nested() -> None:
  """Test class definition nested."""
  code: str = """
class Outer:
    class Inner:
        def __init__(self):
            self.x = 1
    """
  analyze(code)


def test_lambda() -> None:
  """Test lambda expression."""
  code: str = """
f = lambda x: x + 1
    """
  analyze(code)


def test_list_comp() -> None:
  """Test list comprehension."""
  code: str = """
l = [x for x in range(10)]
    """
  analyze(code)
