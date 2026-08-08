"""Test suite for symbol table analysis extra coverage."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer


def analyze(code: str) -> SymbolTableAnalyzer:
  """Analyze code for symbol table."""
  tree = cst.parse_module(code)
  sm = MagicMock()
  analyzer = SymbolTableAnalyzer(sm)
  tree.visit(analyzer)
  return analyzer


def test_missing_symbol_table_coverage():
  """Test missing symbol table coverage."""
  # Test try/except blocks
  code = """
try:
    x = 1
except Exception as e:
    x = 2
finally:
    y = 3
    """
  analyze(code)

  # Test boolean ops
  code = """
x = True and False or True
    """
  analyze(code)

  # Test unary ops
  code = """
x = not True
y = -1
    """
  analyze(code)

  # Test with/async with
  code = """
with open('file.txt') as f:
    x = 1
    """
  analyze(code)


def test_global_scope_access():
  """Test global scope access."""
  code = """
global_var = 1
def func():
    return global_var
    """
  analyze(code)


def test_class_def_nested():
  """Test class definition nested."""
  code = """
class Outer:
    class Inner:
        def __init__(self):
            self.x = 1
    """
  analyze(code)


def test_lambda():
  """Test lambda expression."""
  code = """
f = lambda x: x + 1
    """
  analyze(code)


def test_list_comp():
  """Test list comprehension."""
  code = """
l = [x for x in range(10)]
    """
  analyze(code)
