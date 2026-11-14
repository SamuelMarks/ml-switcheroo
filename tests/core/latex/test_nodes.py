"""Unit tests for MIDL LaTeX semantic CST nodes (`ml_switcheroo.core.latex.nodes`)."""

from ml_switcheroo.core.latex.nodes import (
  ComputeNode,
  DocumentNode,
  EnvironmentNode,
  InputNode,
  LatexNode,
  MacroNode,
  MemoryNode,
  ModelContainer,
  ReturnNode,
  StateOpNode,
  TextNode,
)

_NL = chr(10)


def test_latex_node_base_default_emit() -> None:
  """Verify concrete `LatexNode` subclasses inherit empty `emit()` and `to_latex()` defaults."""

  class MinimalLatexNode(LatexNode):
    """Minimal concrete subclass exercising `LatexNode` base emission."""

  node = MinimalLatexNode()
  assert node.emit() == ""
  assert node.to_latex() == ""
  assert str(node) == ""


def test_text_macro_environment_and_document_emission() -> None:
  """Verify indentation and formatting of `TextNode`, `MacroNode`, `EnvironmentNode`, and `DocumentNode`."""
  text = TextNode(content="some text")
  assert text.emit(0) == "some text"
  assert text.emit(1) == "  some text"

  macro_plain = MacroNode(name="b")
  assert macro_plain.emit() == r"\b"

  macro_full = MacroNode(name="foo", args=["a", "b"], options=["opt1", "opt2"])
  assert macro_full.emit() == r"\foo[opt1, opt2]{a}{b}"

  env = EnvironmentNode(name="env", args=["arg1"], children=[TextNode(content="inner")])
  assert env.emit(0) == _NL.join([r"\begin{env}{arg1}", "  inner", r"\end{env}"])

  doc_empty = DocumentNode(children=[])
  assert doc_empty.emit() == ""

  doc = DocumentNode(children=[TextNode(content="line1"), TextNode(content="line2")])
  assert doc.emit(1) == f"  line1{_NL}  line2"


def test_memory_node_serialization() -> None:
  """Verify `MemoryNode` renders as an `Attribute` LaTeX macro with escaped key-value pairs."""
  node = MemoryNode(node_id="conv", op_type="Conv2d", config={"in": "1", "out": "32"})
  output = node.to_latex()
  assert output == r"\Attribute{conv}{Conv2d}{in=1, out=32}"


def test_input_compute_state_and_return_serialization() -> None:
  """Verify `InputNode`, `ComputeNode`, `StateOpNode`, and `ReturnNode` macro serialization."""
  inp = InputNode(name="x", shape="[B, 32]")
  assert inp.to_latex() == r"\Input{x}{[B, 32]}"

  comp = ComputeNode(node_id="s1", op_type="Flatten", args=["x", "start=1"], shape="[B, 1024]")
  assert comp.to_latex() == r"\Op{s1}{Flatten}{x, start=1}{[B, 1024]}"

  state = StateOpNode(node_id="s2", attribute_id="conv", args=["x"], shape="[B, 32]")
  assert state.to_latex() == r"\StateOp{s2}{conv}{x}{[B, 32]}"

  ret = ReturnNode(target_id="s2")
  assert ret.to_latex() == r"\Return{s2}"


def test_model_container_rendering() -> None:
  """Verify `ModelContainer` renders an indented `DefModel` environment wrapping all child macros."""
  m_conv = MemoryNode("conv", "Conv2d", {"k": "3"})
  m_fc = MemoryNode("fc", "Linear", {"out": "10"})
  i_x = InputNode("x", "[B, 1, 28, 28]")
  op_1 = StateOpNode("s1", "conv", ["x"], "[B, 32]")
  op_2 = ComputeNode("s2", "ReLU", ["s1"], "[B, 32]")
  op_3 = StateOpNode("s3", "fc", ["s2"], "[B, 10]")
  ret = ReturnNode("s3")
  model = ModelContainer(name="Net", children=[m_conv, m_fc, i_x, op_1, op_2, op_3, ret])

  code = model.to_latex()
  assert r"\begin{DefModel}{Net}" in code
  assert r"\end{DefModel}" in code
  lines = code.split(_NL)
  assert lines[1].startswith("  ")
  assert r"\Attribute{conv}{Conv2d}{k=3}" in code
  assert r"\Op{s2}{ReLU}{s1}{[B, 32]}" in code
  assert r"\Return{s3}" in code
  assert repr(model) != ""
