# tests/core/latex/test_parser.py

"""
Tests for the LaTeX DSL Parser.

Verifies:
1. Macro Regex Extraction.
2. Argument Parsing (KV pairs, positional).
3. AST Synthesis (Class Structure, Namespace Injection).
4. End-to-End source conversion.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.latex.parser import LatexParser


@pytest.fixture
def basic_latex():
  return r"""
\documentclass{standalone}
\begin{document}
\begin{DefModel}{SimpleNet}
    \Attribute{fc1}{Linear}{in=10, out=5}
    \Input{data}{[B, 10]}
    
    \StateOp{h1}{fc1}{data}{[B, 5]}
    \Op{act}{ReLU}{h1}{[B, 5]}
    \Return{act}
\end{DefModel}
\end{document}
"""


def test_parser_end_to_end(basic_latex):
  """
  Verify complete transformation from LaTeX to Python AST.
  Checks for `midl` namespace usage.
  """
  parser = LatexParser(basic_latex)
  tree = parser.parse()

  code = tree.code

  # 1. Imports
  assert "import midl" in code

  # 2. Class Name and Inheritance
  assert "class SimpleNet(midl.Module):" in code

  # 3. Init
  # Should use midl.Linear
  assert "self.fc1 = midl.Linear(in=10, out=5)" in code
  assert "super().__init__()" in code

  # 4. Forward
  assert "def forward(self, data):" in code
  assert "h1 = self.fc1(data)" in code
  # Stateless op usage: midl.ReLU
  assert "act = midl.ReLU(h1)" in code
  assert "return act" in code


def test_config_parsing():
  """Verify parsing of key-value config strings."""
  parser = LatexParser("")

  res1 = parser._parse_config_string("a=1, b=2")
  assert res1 == {"a": "1", "b": "2"}

  res2 = parser._parse_config_string("1, 2, k=3")
  assert res2 == {"arg_0": "1", "arg_1": "2", "k": "3"}


def test_complex_args_parsing():
  """Verify parsing of operation arguments."""
  parser = LatexParser("")

  # Mixed positional and keyword-like args in function call
  parsed = parser._parse_arg_list("x, dim=1, keepdim=True")
  assert parsed == ["x", "dim=1", "keepdim=True"]


def test_multiple_attributes(basic_latex):
  """Verify multiple attributes are preserved order-wise."""
  source = r"""
\begin{DefModel}{Multi}
    \Attribute{c1}{Conv}{k=3}
    \Attribute{c2}{Conv}{k=5}
\end{DefModel}
    """
  parser = LatexParser(source)
  code = parser.parse().code

  lines = code.splitlines()
  c1_idx = next(i for i, l in enumerate(lines) if "self.c1" in l)
  c2_idx = next(i for i, l in enumerate(lines) if "self.c2" in l)

  assert c1_idx < c2_idx
  # Check namespacing
  assert "midl.Conv" in code


def test_implicit_flow_synthesis():
  """Verify stateops and normal ops interleave correctly."""
  source = r"""
\begin{DefModel}{Flow}
    \Attribute{l1}{L}{}
    \Input{x}{_}
    \StateOp{a}{l1}{x}{_}
    \Op{b}{Func}{a}{_}
    \StateOp{c}{l1}{b}{_}
\end{DefModel}
    """
  parser = LatexParser(source)
  code = parser.parse().code

  assert "a = self.l1(x)" in code
  assert "b = midl.Func(a)" in code
  assert "c = self.l1(b)" in code
