"""Unit tests for the MIDL LaTeX DSL parser (`ml_switcheroo.core.latex.parser`)."""

from unittest.mock import patch

import libcst as cst
import pytest
from libcst._nodes.base import CSTValidationError

from ml_switcheroo.core.latex.parser import LatexParser
from ml_switcheroo.core.markup.parser import BaseTokenizer, RecursiveDescentParser, Token


@pytest.fixture
def basic_latex() -> str:
  """Provide a standard `DefModel` LaTeX snippet with an attribute, input, state op, functional op, and return.

  Returns:
      Multi-line LaTeX document string defining `SimpleNet`.
  """
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


def test_parser_end_to_end(basic_latex: str) -> None:
  """Verify `LatexParser.parse()` synthesizes a complete `midl.Module` Python class.

  Args:
      basic_latex: Standard `SimpleNet` LaTeX document fixture.
  """
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


def test_config_and_arg_list_parsing() -> None:
  """Verify `_parse_config_string` and `_parse_arg_list` handle positional, keyword, and empty inputs."""
  parser = LatexParser("")
  assert parser._parse_config_string("") == {}
  assert parser._parse_config_string("   ") == {}
  assert parser._parse_config_string("a=1, b=2") == {"a": "1", "b": "2"}
  assert parser._parse_config_string("1, 2, k=3") == {"arg_0": "1", "arg_1": "2", "k": "3"}
  assert parser._parse_config_string("k=,k2=v") == {"k": "", "k2": "v"}

  assert parser._parse_arg_list("") == []
  assert parser._parse_arg_list("   ") == []
  assert parser._parse_arg_list("x, dim=1, keepdim=True") == ["x", "dim=1", "keepdim=True"]


def test_safe_value_node_variants() -> None:
  """Verify `_safe_value_node` parses ellipsis, Python literals, and falls back to `cst.Name` on syntax error."""
  parser = LatexParser("")
  assert isinstance(parser._safe_value_node("..."), cst.Ellipsis)
  assert isinstance(parser._safe_value_node("1"), cst.Integer)
  assert isinstance(parser._safe_value_node("1.23"), cst.Float)
  assert isinstance(parser._safe_value_node("'hello'"), cst.SimpleString)
  assert isinstance(parser._safe_value_node("[1, 2]"), cst.List)
  assert isinstance(parser._safe_value_node("True"), cst.Name)

  with patch("libcst.parse_expression", side_effect=cst.ParserSyntaxError("err", lines=[""], raw_line=0, raw_column=0)):
    fallback = parser._safe_value_node("valid_name")
    assert isinstance(fallback, cst.Name)
    assert fallback.value == "valid_name"


def test_create_call_dotted_and_positional_or_keyword_args() -> None:
  """Verify `_create_call` constructs simple and dotted `cst.Call` expressions from config and args lists."""
  parser = LatexParser("")

  simple = parser._create_call("SimpleFunc")
  assert isinstance(simple.func, cst.Name)
  assert simple.func.value == "SimpleFunc"
  assert list(simple.args) == []

  dotted_cfg = parser._create_call("func.subfunc", config={"arg_0": "val", "k": "1"})
  assert isinstance(dotted_cfg.func, cst.Attribute)
  assert len(dotted_cfg.args) == 2
  assert dotted_cfg.args[0].keyword is None
  assert cst.ensure_type(dotted_cfg.args[0].value, cst.Name).value == "val"
  assert cst.ensure_type(dotted_cfg.args[1].keyword, cst.Name).value == "k"

  dotted_list = parser._create_call("func.subfunc", args_list=["pos", "kw=v"])
  assert len(dotted_list.args) == 2
  assert dotted_list.args[0].keyword is None
  assert cst.ensure_type(dotted_list.args[1].keyword, cst.Name).value == "kw"


def test_multiple_attributes_and_implicit_flow_without_return() -> None:
  """Verify attribute ordering in `__init__` and `pass` fallback when `Return` is omitted."""
  source = r"""
\begin{DefModel}{Flow}
    \Attribute{c1}{Conv}{k=3}
    \Attribute{c2}{Conv}{k=5}
    \Input{x}{_}
    \StateOp{a}{c1}{x}{_}
    \Op{b}{Func}{a}{_}
    \StateOp{c}{c2}{b}{_}
\end{DefModel}
"""
  code = LatexParser(source).parse().code
  lines = code.splitlines()
  c1_idx = next(i for i, line in enumerate(lines) if "self.c1" in line)
  c2_idx = next(i for i, line in enumerate(lines) if "self.c2" in line)
  assert c1_idx < c2_idx
  assert "a = self.c1(x)" in code
  assert "b = midl.Func(a)" in code
  assert "c = self.c2(b)" in code
  assert "pass" in code


def test_nested_braces_brackets_comments_and_unknown_environments() -> None:
  """Verify balanced brace/bracket consumption, comments, escaped percents, and unknown environments."""
  doc = r"""
% Leading comment
\% Escaped percent
\UnknownOutside[opt[nested]]{{Nested{Inside}Name}}
\begin{OtherEnv}
\end{OtherEnv}
\begin{DefModel}
\begin{enumerate}
\end{enumerate}
\Attribute{conv1}{Conv2d}{}
\StateOp{s1}{conv1}{}{}
\Op{s2}{Flatten}{"a", {"nested": {1: 2}}}{shape}
\1
\end{DefModel}
"""
  mod = LatexParser(doc).parse()
  assert "class GeneratedModel(midl.Module):" in mod.code
  assert "self.conv1 = midl.Conv2d()" in mod.code
  assert "s2 = midl.Flatten" in mod.code

  invalid_name_doc = r"\begin{DefModel}{MyModel{Nested}}\end{DefModel}"
  with pytest.raises(CSTValidationError):
    LatexParser(invalid_name_doc).parse()

  null_byte = chr(0)
  skipped_null = LatexParser(r"\begin{DefModel}{ValidModel}" + null_byte + r"\Return{x}\end{DefModel}").parse()
  assert "class ValidModel(midl.Module):" in skipped_null.code


def test_markup_tokenizer_and_recursive_descent_primitives() -> None:
  """Verify `BaseTokenizer` `SKIP` handling and `RecursiveDescentParser` token cursor operations."""
  tokenizer = BaseTokenizer([("SKIP", r"\s+"), ("WORD", r"\w+")])
  tokens = tokenizer.tokenize("hello   world")
  assert [t.value for t in tokens] == ["hello", "world"]

  tok = Token("TEST", "val")
  assert repr(tok) == "Token(TEST, 'val')"
  rd = RecursiveDescentParser([tok])
  assert rd.match("OTHER") is False
  assert rd.match("TEST") is True
  assert rd.advance() is None
  with pytest.raises(ValueError, match="Expected TEST"):
    rd.expect("TEST")
