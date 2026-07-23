"""Test suite for the Latex Nodes module."""

from ml_switcheroo.core.latex.nodes import ModelContainer, MemoryNode, InputNode, ComputeNode, StateOpNode, ReturnNode


def test_memory_node_serialization():
  """Verifies the behavior of memory node serialization."""
  node = MemoryNode(node_id="conv", op_type="Conv2d", config={"in": "1", "out": "32"})
  output = node.to_latex()
  assert "\\Attribute{conv}{Conv2d}" in output
  assert "in=1" in output
  assert "out=32" in output


def test_input_node_serialization():
  """Verifies the behavior of input node serialization."""
  node = InputNode(name="x", shape="[B, 32]")
  output = node.to_latex()
  assert output == "\\Input{x}{[B, 32]}"


def test_compute_node_serialization():
  """Computes node serialization."""
  node = ComputeNode(node_id="s1", op_type="Flatten", args=["x", "start=1"], shape="[B, 1024]")
  output = node.to_latex()
  assert output == "\\Op{s1}{Flatten}{x, start=1}{[B, 1024]}"


def test_state_op_node_serialization():
  """Verifies the behavior of state op node serialization."""
  node = StateOpNode(node_id="s2", attribute_id="conv", args=["x"], shape="[B, 32]")
  output = node.to_latex()
  assert output == "\\StateOp{s2}{conv}{x}{[B, 32]}"


def test_return_node_serialization():
  """Verifies the behavior of return node serialization."""
  node = ReturnNode(target_id="s2")
  assert node.to_latex() == "\\Return{s2}"


def test_model_container_rendering():
  """Verifies the behavior of model container rendering."""
  m_conv = MemoryNode("conv", "Conv2d", {"k": "3"})
  m_fc = MemoryNode("fc", "Linear", {"out": "10"})
  i_x = InputNode("x", "[B, 1, 28, 28]")
  op_1 = StateOpNode("s1", "conv", ["x"], "[B, 32]")
  op_2 = ComputeNode("s2", "ReLU", ["s1"], "[B, 32]")
  op_3 = StateOpNode("s3", "fc", ["s2"], "[B, 10]")
  ret = ReturnNode("s3")
  model = ModelContainer(name="Net", children=[m_conv, m_fc, i_x, op_1, op_2, op_3, ret])
  code = model.to_latex()
  assert "\\begin{DefModel}{Net}" in code
  assert "\\end{DefModel}" in code
  lines = code.split("\n")
  assert lines[1].startswith("  ")
  assert "\\Attribute{conv}{Conv2d}{k=3}" in code
  assert "\\Op{s2}{ReLU}{s1}{[B, 32]}" in code
  assert "\\Return{s3}" in code
