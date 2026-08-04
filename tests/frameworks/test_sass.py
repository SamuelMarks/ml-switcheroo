"""Test suite for the Sass module."""

from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_sass_adapter_init():
  """Verifies the behavior of SASS adapter initialization."""
  adapter = SassAdapter()
  assert adapter.display_name == "NVIDIA SASS"
  assert adapter.ui_priority == 150
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_sass_properties():
  """Verifies the behavior of SASS properties."""
  adapter = SassAdapter()
  assert adapter.import_alias == ("sass", "asm")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base is None
  config = adapter.test_config
  assert "// SASS Header" in config["import"]
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
  assert "FADD" in examples["tier1_math"]


def test_sass_missing_coverage():
  """Verifies the remaining untested methods of SASS adapter."""
  adapter = SassAdapter()

  # Traits
  assert adapter.plugin_traits is not None

  # Device & RNG
  assert adapter.get_device_syntax("cuda") == "// Target Device: cuda"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""

  # Serialization
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file.pt") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "// Weights loading not supported in SASS adapter"
  assert adapter.get_tensor_to_numpy_expr("my_var") == "my_var"
  assert adapter.get_weight_save_code("state", "path") == "// Weights saving not supported in SASS adapter"

  # Documentation
  assert adapter.get_doc_url("my_api") is None

  # Convert
  assert adapter.convert(123) == "123"

  # Graph parsing
  code_loop = """
  // comment
  FADD R1, R1, R2
  ISETP.LT.AND P0, PT, R5, 128, PT;
  FFMA R1, R3, R5, R1
  L_LABEL:
  """
  graph_loop = adapter.parse_sass_to_graph(code_loop)
  nodes_loop = list(graph_loop.nodes.values())
  assert len(nodes_loop) == 2
  assert nodes_loop[0].op_type == "LoopControl"
  assert nodes_loop[1].op_type == "Conv2d"

  code_no_loop = """
  FFMA R1, R3, R5, R1
  """
  graph_no_loop = adapter.parse_sass_to_graph(code_no_loop)
  nodes_no_loop = list(graph_no_loop.nodes.values())
  assert len(nodes_no_loop) == 1
  assert nodes_no_loop[0].op_type == "Linear"
