"""Unit tests for `RdnaAdapter` (`ml_switcheroo.frameworks.rdna`)."""

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_rdna_adapter_properties_syntax_and_wiring() -> None:
  """Verify `RdnaAdapter` metadata, traits, device/serialization/weight syntax, and doc URL."""
  adapter = RdnaAdapter()
  assert adapter.display_name == "AMD RDNA"
  assert adapter.ui_priority == 151
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST
  assert adapter.target_arch == "gfx1030"
  assert adapter.import_alias == ("rdna", "asm")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.structural_traits.module_base is None
  assert isinstance(adapter.plugin_traits, PluginTraits)
  assert "; RDNA Header" in adapter.test_config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  assert isinstance(adapter.definitions, dict)
  assert adapter.specifications == {}

  assert adapter.get_device_syntax("gpu") == "; Target Device: gpu"
  assert adapter.get_device_syntax("cpu") == "; Target Device: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file.pt", "obj") == ""
  assert adapter.get_serialization_syntax("load", "file.pt") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "; Weights loading not supported in RDNA adapter"
  assert adapter.get_tensor_to_numpy_expr("my_var") == "my_var"
  assert adapter.get_weight_save_code("state", "path") == "; Weights saving not supported in RDNA adapter"
  assert adapter.get_doc_url("my_api") == "https://gpuopen.com/learn/rdna-performance-guide/?q=my_api"
  assert adapter.convert(123) == "123"
  assert adapter.convert("data") == "data"

  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}

  ex = adapter.get_tiered_examples()
  assert {"tier1_math", "tier2_neural_simple", "tier3_extras"} <= set(ex.keys())
  assert "v_add_f32" in ex["tier1_math"]


def test_rdna_parse_rdna_to_graph_loops_labels_and_linear() -> None:
  """Verify `parse_rdna_to_graph` across empty comments, consecutive labels, loop CFGs, and loop-free MACs."""
  nl = chr(10)
  adapter = RdnaAdapter()
  assert len(adapter.parse_rdna_to_graph("; just a comment").nodes) == 0
  assert len(adapter.parse_rdna_to_graph(f"   {nl}// comment{nl}s_branch").nodes) == 0

  label_graph = adapter.parse_rdna_to_graph(
    """L_123:
BB0_1:
v_mac_f32 v0, v1, v2
"""
  )
  label_nodes = list(label_graph.nodes.values())
  assert len(label_nodes) == 1
  assert label_nodes[0].op_type == "Linear"

  loop_code = """
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
  loop_graph = adapter.parse_rdna_to_graph(loop_code)
  assert loop_graph.name == "Model"
  assert {n.op_type for n in loop_graph.nodes.values()} == {"LoopControl", "Conv2d"}
