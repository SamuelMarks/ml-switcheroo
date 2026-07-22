"""Test suite for the Latex Parser module."""

import pytest
from ml_switcheroo.core.latex.parser import LatexParser


@pytest.fixture
def basic_latex():
  """Provides a mock basic LaTeX for testing."""
  return "\n\\documentclass{standalone}\n\\begin{document}\n\\begin{DefModel}{SimpleNet}\n    \\Attribute{fc1}{Linear}{in=10, out=5}\n    \\Input{data}{[B, 10]}\n\n    \\StateOp{h1}{fc1}{data}{[B, 5]}\n    \\Op{act}{ReLU}{h1}{[B, 5]}\n    \\Return{act}\n\\end{DefModel}\n\\end{document}\n"


def test_parser_end_to_end(basic_latex):
  """Verifies the behavior of parser end to end."""
  parser = LatexParser(basic_latex)
  tree = parser.parse()
  code = tree.code
  assert "import midl" in code
  assert "class SimpleNet(midl.Module):" in code
  assert "self.fc1 = midl.Linear(in=10, out=5)" in code
  assert "super().__init__()" in code
  assert "def forward(self, data):" in code
  assert "h1 = self.fc1(data)" in code
  assert "act = midl.ReLU(h1)" in code
  assert "return act" in code


def test_config_parsing():
  """Verifies the behavior of configuration parsing."""
  parser = LatexParser("")
  res1 = parser._parse_config_string("a=1, b=2")
  assert res1 == {"a": "1", "b": "2"}
  res2 = parser._parse_config_string("1, 2, k=3")
  assert res2 == {"arg_0": "1", "arg_1": "2", "k": "3"}


def test_complex_args_parsing():
  """Verifies the behavior of complex arguments parsing."""
  parser = LatexParser("")
  parsed = parser._parse_arg_list("x, dim=1, keepdim=True")
  assert parsed == ["x", "dim=1", "keepdim=True"]


def test_multiple_attributes(basic_latex):
  """Verifies the behavior of multiple attributes."""
  source = (
    "\n\\begin{DefModel}{Multi}\n    \\Attribute{c1}{Conv}{k=3}\n    \\Attribute{c2}{Conv}{k=5}\n\\end{DefModel}\n    "
  )
  parser = LatexParser(source)
  code = parser.parse().code
  lines = code.splitlines()
  c1_idx = next((i for (i, line) in enumerate(lines) if "self.c1" in line))
  c2_idx = next((i for (i, line) in enumerate(lines) if "self.c2" in line))
  assert c1_idx < c2_idx
  assert "midl.Conv" in code


def test_implicit_flow_synthesis():
  """Verifies the behavior of implicit flow synthesis."""
  source = "\n\\begin{DefModel}{Flow}\n    \\Attribute{l1}{L}{}\n    \\Input{x}{_}\n    \\StateOp{a}{l1}{x}{_}\n    \\Op{b}{Func}{a}{_}\n    \\StateOp{c}{l1}{b}{_}\n\\end{DefModel}\n    "
  parser = LatexParser(source)
  code = parser.parse().code
  assert "a = self.l1(x)" in code
  assert "b = midl.Func(a)" in code
  assert "c = self.l1(b)" in code
