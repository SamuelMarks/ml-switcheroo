"""Test module."""

from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_sass_missing_methods():
  """Test function."""
  adapter = SassAdapter()

  assert adapter.get_device_syntax("cpu") == "// Target Device: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file", "obj") == ""
  assert adapter.get_serialization_syntax("load", "file") == ""
  assert adapter.get_serialization_syntax("invalid", "file") == ""
  assert adapter.get_serialization_syntax("save", "file", None) == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "// Weights loading not supported in SASS adapter"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "// Weights saving not supported in SASS adapter"

  assert isinstance(adapter.plugin_traits, PluginTraits)
  assert adapter.convert("data") == "data"

  defs = adapter.definitions
  assert isinstance(defs, dict)

  ex = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier1_math" in ex
  pass

  adapter.apply_wiring({})
  assert adapter.get_doc_url("api") is None


def test_sass_parse_sass_to_graph():
  """Test function."""
  adapter = SassAdapter()

  # Test empty graph
  empty_graph = adapter.parse_sass_to_graph("// just a comment")
  assert len(empty_graph.nodes) == 0

  # Code with a loop
  # entry block (implicit) falls through to L_LOOP
  # L_LOOP has no FFMA, so it gets LoopControl
  # L_BODY has FFMA and branches back to L_LOOP
  code = """
    // comment
    entry:
    MOV R0, R0

    L_LOOP:
    ISETP.LT.AND P1, PT, R0, 0x10, PT
    BRA L_BODY

    L_BODY:
    FFMA R1, R2, R3, R4
    BRA L_LOOP
    """

  graph1 = adapter.parse_sass_to_graph(code)
  assert graph1.name == "Model"
  nodes1 = list(graph1.nodes.values())
  # L_LOOP is in loop, no FFMA -> LoopControl
  # L_BODY is in loop, has FFMA -> Conv2d
  # entry is not in loop, no FFMA -> no node added
  assert len(nodes1) == 2
  op_types = {n.op_type for n in nodes1}
  assert "LoopControl" in op_types
  assert "Conv2d" in op_types

  # code without loop
  code_no_loop = """
    FFMA R1, R2, R3, R4
    """
  graph2 = adapter.parse_sass_to_graph(code_no_loop)
  nodes2 = list(graph2.nodes.values())
  assert len(nodes2) == 1
  assert nodes2[0].op_type == "Linear"
