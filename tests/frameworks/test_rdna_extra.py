"""Test module."""

from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_rdna_missing_methods():
  """Test function."""
  adapter = RdnaAdapter()

  assert adapter.get_device_syntax("cpu") == "; Target Device: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file", "obj") == ""
  assert adapter.get_serialization_syntax("load", "file") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "; Weights loading not supported in RDNA adapter"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "; Weights saving not supported in RDNA adapter"

  assert isinstance(adapter.plugin_traits, PluginTraits)
  assert adapter.convert("data") == "data"
  assert adapter.ui_priority == 151

  defs = adapter.definitions
  assert isinstance(defs, dict)

  ex = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural_simple" in ex
  assert "tier3_extras" in ex
  pass

  adapter.apply_wiring({})
  assert "gpuopen.com" in adapter.get_doc_url("api")


def test_rdna_parse_rdna_to_graph():
  """Test parsing RDNA code to graph."""
  adapter = RdnaAdapter()

  # Test empty graph
  empty_graph = adapter.parse_rdna_to_graph("; just a comment")
  assert len(empty_graph.nodes) == 0

  # Code with a loop
  code = """
    ; comment
    entry:
    s_mov_b32 s0, s0

    L_LOOP:
    s_cmp_eq_u32 s1, 0
    s_cbranch_vccnz L_BODY

    L_BODY:
    v_mac_f32 v1, v2, v3
    s_branch L_LOOP
    """

  graph1 = adapter.parse_rdna_to_graph(code)
  assert graph1.name == "Model"
  nodes1 = list(graph1.nodes.values())
  assert len(nodes1) == 2
  op_types = {n.op_type for n in nodes1}
  assert "LoopControl" in op_types
  assert "Conv2d" in op_types

  # code without loop
  code_no_loop = """
    v_mac_f32 v1, v2, v3
    """
  graph2 = adapter.parse_rdna_to_graph(code_no_loop)
  nodes2 = list(graph2.nodes.values())
  assert len(nodes2) == 1
  assert nodes2[0].op_type == "Linear"
