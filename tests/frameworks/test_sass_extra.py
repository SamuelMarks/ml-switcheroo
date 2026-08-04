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

  # 293-323 lines
  code = """
    // comment

    L_LABEL:
    ISETP.LT.AND P1, PT, R0, 0x10, PT
    FFMA R1, R2, R3, R4
    """

  # code with loop
  graph1 = adapter.parse_sass_to_graph(code)
  assert graph1.name == "Model"
  nodes1 = list(graph1.nodes.values())
  assert len(nodes1) == 2
  assert nodes1[0].op_type == "LoopControl"
  assert nodes1[1].op_type == "Conv2d"

  # code without loop
  code_no_loop = """
    FFMA R1, R2, R3, R4
    """
  graph2 = adapter.parse_sass_to_graph(code_no_loop)
  nodes2 = list(graph2.nodes.values())
  assert len(nodes2) == 1
  assert nodes2[0].op_type == "Linear"
