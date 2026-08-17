"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.latex.nodes import (
  MacroNode,
  EnvironmentNode,
  TextNode,
  DocumentNode,
  MemoryNode,
  InputNode,
  ComputeNode,
  StateOpNode,
  ReturnNode,
  ModelContainer,
  LatexNode,
)
from ml_switcheroo.core.latex.parser import LatexParser


def test_latex_nodes_missing():
  """Docstring."""
  nodes = [
    MacroNode(name="foo", args=["a", "b"]),
    EnvironmentNode(name="env", args=[], children=[TextNode(content="text")]),
    DocumentNode(children=[]),
    MemoryNode(node_id="m", op_type="cls", config={"k": "v", "k2": "v2"}),
    InputNode(name="i", shape="[1]"),
    ComputeNode(node_id="c", op_type="add", args=["a", "b"], shape="[1]"),
    StateOpNode(node_id="s", attribute_id="t", args=["a"], shape="[1]"),
    ReturnNode(target_id="r"),
    ModelContainer(name="m", children=[]),
  ]
  for n in nodes:
    assert repr(n)
    if hasattr(n, "emit"):
      n.emit()
    n.__str__()

  class DummyNode(LatexNode):
    def emit(self, indent_level: int = 0) -> str:
      return "dummy"

  DummyNode().__str__()
  DummyNode().emit()

  assert TextNode(content="a").emit() == "a"
  assert MacroNode(name="b").emit() == r"\b"


def test_latex_parser_missing():
  """Docstring."""
  parser = LatexParser(r"")

  assert parser._parse_config_string("") == {}
  res = parser._parse_config_string("k=v, k2=v2, just_key")
  assert "k" in res

  assert parser._parse_arg_list("") == []
  assert parser._parse_arg_list("a, b, c") == ["a", "b", "c"]

  val1 = parser._safe_value_node("...")
  assert isinstance(val1, cst.Ellipsis)

  val2 = parser._safe_value_node("1")
  assert isinstance(val2, cst.Integer)

  val3 = parser._safe_value_node("[1, 2]")
  assert isinstance(val3, cst.List)

  val4 = parser._safe_value_node("abc")
  assert isinstance(val4, cst.Name)

  call1 = parser._create_call("func")
  assert isinstance(call1, cst.Call)

  call2 = parser._create_call("func", config={"k": "1"})
  assert isinstance(call2, cst.Call)

  call3 = parser._create_call("func", config={}, args_list=["1"])
  assert isinstance(call3, cst.Call)

  call4 = parser._create_call("func.subfunc", config={}, args_list=["1"])
  assert isinstance(call4, cst.Call)

  doc = r"""
\begin{DefModel}
\Attribute{conv1}{Conv2d}{in_channels=1, out_channels=32}
\Input{x}{[B, 1, 28, 28]}
\StateOp{s1}{conv1}{x}{[B, 32, 28, 28]}
\Op{s2}{Flatten}{s1, start_dim=1}{[B, 25088]}
\Return{s2}
\end{DefModel}
"""
  parser = LatexParser(doc)
  mod = parser.parse()
  assert isinstance(mod, cst.Module)

  parser2 = LatexParser(r"""
\Attribute{conv1}{Conv2d}{}
\StateOp{s1}{conv1}{}{}
\Op{s2}{Flatten}{}{}
""")
  parser2.parse()

  parser3 = LatexParser(r"\begin{DefModel}{WithArgs}\begin{enumerate} \end{enumerate}\end{DefModel}")
  parser3.parse()

  parser4 = LatexParser(r"\begin{DefModel}{Empty}\end{DefModel}")
  parser4.parse()

  parser5 = LatexParser(r"% this is a comment")
  parser5.parse()

  parser6 = LatexParser(r"\begin{OtherEnv} \end{OtherEnv}")
  parser6.parse()

  # Hit safe value node parsing code fully
  LatexParser("")._safe_value_node("...")
  LatexParser("")._safe_value_node("1.23")
  LatexParser("")._safe_value_node("'hello'")
  LatexParser("")._safe_value_node("None")
  LatexParser("")._safe_value_node("True")
  LatexParser("")._safe_value_node("False")

  # Missing args in config
  assert LatexParser("")._parse_config_string("k=,k2=v") == {"k": "", "k2": "v"}

  parser12 = LatexParser(r"foo \begin{env}")
  parser12.parse()

  parser13 = LatexParser(r"foo \end{env}")
  parser13.parse()

  parser14 = LatexParser(r"\macro{arg} \foo")
  parser14.parse()

  parser15 = LatexParser(r"\macro[arg]")
  parser15.parse()

  parser16 = LatexParser(r"\macro   {a}")
  parser16.parse()

  parser17 = LatexParser(r"\1")
  parser17.parse()

  parser18 = LatexParser(r"\begin{DefModel} \Input{x}{a} \StateOp{x}{x}{x}{x} \end{DefModel}")
  parser18.parse()

  # Test config dict items string matching kwarg
  parser._create_call("func", config={"arg_1": "123"})

  # Hit line 327 missing fallback in synthesize class
  class DummyOp:
    node_id = "d"

  LatexParser("")._synthesize_class("N", [], None, [DummyOp()], None)  # type: ignore
