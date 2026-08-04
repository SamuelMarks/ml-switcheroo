"""Test suite for the Rdna module."""

from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_rdna_adapter_init():
  """Verifies the behavior of RDNA adapter initialization."""
  adapter = RdnaAdapter()
  assert adapter.display_name == "AMD RDNA"
  assert adapter.ui_priority == 151
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST
  assert adapter.target_arch == "gfx1030"


def test_rdna_properties():
  """Verifies the behavior of RDNA properties."""
  adapter = RdnaAdapter()
  assert adapter.import_alias == ("rdna", "asm")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base is None
  config = adapter.test_config
  assert "; RDNA Header" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs = adapter.definitions
  assert isinstance(defs, dict)
  specs = adapter.specifications
  assert specs == {}
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "v_add_f32" in examples["tier1_math"]


def test_rdna_missing_coverage():
  """Verifies the remaining untested methods of RDNA adapter."""
  adapter = RdnaAdapter()

  # Traits
  assert adapter.plugin_traits is not None

  # Device & RNG
  assert adapter.get_device_syntax("gpu") == "; Target Device: gpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""

  # Serialization
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file.pt") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "; Weights loading not supported in RDNA adapter"
  assert adapter.get_tensor_to_numpy_expr("my_var") == "my_var"
  assert adapter.get_weight_save_code("state", "path") == "; Weights saving not supported in RDNA adapter"

  # Documentation
  assert adapter.get_doc_url("my_api") == "https://gpuopen.com/learn/rdna-performance-guide/?q=my_api"

  # Convert
  assert adapter.convert(123) == "123"

  # Graph parsing
  code = """
  // comment
  v_add_f32 v0, v1, v2
  s_cbranch_vccnz label
  v_mac_f32 v3, v4, v5
  """
  graph_loop = adapter.parse_rdna_to_graph(code)
  nodes_loop = list(graph_loop.nodes.values())
  assert len(nodes_loop) == 2
  assert nodes_loop[0].op_type == "LoopControl"
  assert nodes_loop[1].op_type == "Conv2d"

  code_no_loop = """
  v_fmac_f32 v3, v4, v5
  """
  graph_no_loop = adapter.parse_rdna_to_graph(code_no_loop)
  nodes_no_loop = list(graph_no_loop.nodes.values())
  assert len(nodes_no_loop) == 1
  assert nodes_no_loop[0].op_type == "Linear"
