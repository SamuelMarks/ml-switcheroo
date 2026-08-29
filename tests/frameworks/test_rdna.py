"""Test suite for the Rdna module."""

import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_rdna_adapter_init() -> None:
  """Verifies the behavior of RDNA adapter initialization."""
  adapter = RdnaAdapter()
  assert adapter.display_name == "AMD RDNA"
  assert adapter.ui_priority == 151
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST
  assert adapter.target_arch == "gfx1030"


def test_rdna_properties() -> None:
  """Verifies the behavior of RDNA properties."""
  adapter = RdnaAdapter()
  assert adapter.import_alias == ("rdna", "asm")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base is None
  config: dict[str, typing.Any] = adapter.test_config
  assert "; RDNA Header" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)
  specs: typing.Any = adapter.specifications
  assert specs == {}
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "v_add_f32" in examples["tier1_math"]


def test_rdna_missing_coverage() -> None:
  """Docstring."""
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
  code: str = """
  ; comment
  BB0_1:
  v_add_f32 v0, v1, v2
  v_mac_f32 v3, v4, v5
  s_cbranch_vccnz BB0_1
  BB0_2:
  """
  graph_loop: typing.Any = adapter.parse_rdna_to_graph(code)
  nodes_loop: list[typing.Any] = list(graph_loop.nodes.values())
  assert any(n.op_type == "Conv2d" for n in nodes_loop)

  code_no_loop: str = """
  v_mac_f32 v3, v4, v5
  """
  graph_no_loop: typing.Any = adapter.parse_rdna_to_graph(code_no_loop)
  nodes_no_loop: list[typing.Any] = list(graph_no_loop.nodes.values())
  assert len(nodes_no_loop) == 1
  assert nodes_no_loop[0].op_type == "Linear"


# --- Merged from test_rdna_extra.py ---


def test_rdna_missing_methods() -> None:
  """Docstring."""
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

  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)

  ex: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural_simple" in ex
  assert "tier3_extras" in ex
  pass

  adapter.apply_wiring({})
  url: typing.Optional[str] = adapter.get_doc_url("api")
  assert url is not None
  assert "gpuopen.com" in url


def test_rdna_parse_rdna_to_graph() -> None:
  """Docstring."""
  adapter = RdnaAdapter()

  # Test empty graph
  empty_graph: typing.Any = adapter.parse_rdna_to_graph("; just a comment")
  assert len(empty_graph.nodes) == 0

  # Code with a loop
  code: str = """
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

  graph1: typing.Any = adapter.parse_rdna_to_graph(code)
  assert graph1.name == "Model"
  nodes1: list[typing.Any] = list(graph1.nodes.values())
  assert len(nodes1) == 2
  op_types: set[str] = {n.op_type for n in nodes1}
  assert "LoopControl" in op_types
  assert "Conv2d" in op_types

  # code without loop
  code_no_loop: str = """
    v_mac_f32 v1, v2, v3
    """
  graph2: typing.Any = adapter.parse_rdna_to_graph(code_no_loop)
  nodes2: list[typing.Any] = list(graph2.nodes.values())
  assert len(nodes2) == 1
  assert nodes2[0].op_type == "Linear"
