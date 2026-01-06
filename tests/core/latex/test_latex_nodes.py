"""
Tests for MIDL Semantic Nodes.

Verifies:
1.  Data structure integrity.
2.  Latex serialization logic (Macro generation).
3.  Container nesting structure.
"""

from ml_switcheroo.core.latex.nodes import (
  ModelContainer,
  MemoryNode,
  InputNode,
  ComputeNode,
  StateOpNode,
  ReturnNode,
)


def test_memory_node_serialization():
  """Verify \\Attribute rendering."""
  # self.conv = nn.Conv2d(in=1, out=32)
  node = MemoryNode(node_id="conv", op_type="Conv2d", config={"in": "1", "out": "32"})
  output = node.to_latex()

  assert r"\Attribute{conv}{Conv2d}" in output
  assert "in=1" in output
  assert "out=32" in output


def test_input_node_serialization():
  """Verify \\Input rendering."""
  node = InputNode(name="x", shape="[B, 32]")
  output = node.to_latex()
  assert output == r"\Input{x}{[B, 32]}"


def test_compute_node_serialization():
  """Verify \\Op rendering."""
  # torch.flatten(x, start=1)
  node = ComputeNode(node_id="s1", op_type="Flatten", args=["x", "start=1"], shape="[B, 1024]")
  output = node.to_latex()

  # \Op{ID}{Type}{Args}{Shape}
  assert output == r"\Op{s1}{Flatten}{x, start=1}{[B, 1024]}"


def test_state_op_node_serialization():
  """Verify \\StateOp rendering."""
  # self.conv(x)
  node = StateOpNode(node_id="s2", attribute_id="conv", args=["x"], shape="[B, 32]")
  output = node.to_latex()

  # \StateOp{ID}{Attr}{Args}{Shape}
  assert output == r"\StateOp{s2}{conv}{x}{[B, 32]}"


def test_return_node_serialization():
  """Verify \\Return rendering."""
  node = ReturnNode(target_id="s2")
  assert node.to_latex() == r"\Return{s2}"


def test_model_container_rendering():
  """
  Verify full model structure with indentation.
  """
  # Build complete graph
  m_conv = MemoryNode("conv", "Conv2d", {"k": "3"})
  m_fc = MemoryNode("fc", "Linear", {"out": "10"})

  i_x = InputNode("x", "[B, 1, 28, 28]")

  op_1 = StateOpNode("s1", "conv", ["x"], "[B, 32]")
  op_2 = ComputeNode("s2", "ReLU", ["s1"], "[B, 32]")
  op_3 = StateOpNode("s3", "fc", ["s2"], "[B, 10]")

  ret = ReturnNode("s3")

  model = ModelContainer(name="Net", children=[m_conv, m_fc, i_x, op_1, op_2, op_3, ret])

  code = model.to_latex()

  # Check Environment
  assert r"\begin{DefModel}{Net}" in code
  assert r"\end{DefModel}" in code

  # Check Indentation exists (at least 4 spaces)
  lines = code.split("\n")
  assert lines[1].startswith("    ")

  # Check content presence
  assert r"\Attribute{conv}{Conv2d}{k=3}" in code
  assert r"\Op{s2}{ReLU}{s1}{[B, 32]}" in code
  assert r"\Return{s3}" in code
