"""Tests for assembly-level framework adapters."""

from unittest.mock import MagicMock, patch

from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.frameworks.sass import SassAdapter


def test_rdna_adapter_properties() -> None:
  """Docstring."""
  adapter: RdnaAdapter = RdnaAdapter()

  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers
  adapter.structural_traits
  adapter.plugin_traits
  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.rdna.load_definitions", return_value={"test": MagicMock()}):
    adapter.definitions

  adapter.specifications
  adapter.convert([1, 2])
  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path", "obj")
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("v_add_f32")
  adapter.get_tiered_examples()


def test_sass_adapter_properties() -> None:
  """Docstring."""
  adapter: SassAdapter = SassAdapter()

  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers
  adapter.structural_traits
  adapter.plugin_traits
  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.sass.load_definitions", return_value={"test": MagicMock()}):
    adapter.definitions

  adapter.specifications
  adapter.convert([1, 2])
  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path", "obj")
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("FADD")
  adapter.get_tiered_examples()


def test_rdna_parse_to_graph() -> None:
  """Docstring."""
  adapter: RdnaAdapter = RdnaAdapter()
  code: str = """
    label_start:
    s_mov_b32 s0, 1
    v_add_f32 v0, v1, v2
    v_fmac_f32 v0, v1, v2
    s_cmp_eq_u32 s0, 0
    s_cbranch_scc1 label_end
    label_loop:
    s_add_u32 s0, s0, 1
    v_fmac_f32 v0, v1, v2
    FFMA R0, R1, R2, R3
    s_branch label_loop
    label_end:
    s_endpgm
    """
  graph: LogicalGraph = adapter.parse_rdna_to_graph(code)
  assert graph is not None


def test_sass_parse_to_graph() -> None:
  """Docstring."""
  adapter: SassAdapter = SassAdapter()
  code: str = """
    L_1:
    MOV R0, R1
    FADD R2, R3, R4
    FFMA R0, R1, R2, R3
    ISETP.EQ.AND P0, PT, R0, R1
    @P0 BRA L_3
    L_2:
    IADD3 R0, R0, 1
    FFMA R0, R1, R2, R3
    BRA L_2
    L_3:
    EXIT
    """
  graph: LogicalGraph = adapter.parse_sass_to_graph(code)
  assert graph is not None


def test_rdna_fmac_no_loop() -> None:
  """Docstring."""
  adapter: RdnaAdapter = RdnaAdapter()
  code: str = "v_fmac_f32 v0, v1, v2"
  adapter.parse_rdna_to_graph(code)


def test_sass_ffma_no_loop() -> None:
  """Docstring."""
  adapter: SassAdapter = SassAdapter()
  code: str = "FFMA R0, R1, R2, R3"
  adapter.parse_sass_to_graph(code)


def test_asm_empty_parsing() -> None:
  """Docstring."""
  adapter1: RdnaAdapter = RdnaAdapter()
  adapter1.parse_rdna_to_graph("")
  adapter1.parse_rdna_to_graph("// comment")

  adapter2: SassAdapter = SassAdapter()
  adapter2.parse_sass_to_graph("")
  adapter2.parse_sass_to_graph("/* comment */")


def test_rdna_loop_no_fmac() -> None:
  """Docstring."""
  adapter: RdnaAdapter = RdnaAdapter()
  code: str = """
    label_loop:
    s_add_u32 s0, s0, 1
    s_branch label_loop
    """
  adapter.parse_rdna_to_graph(code)


def test_sass_loop_no_fmac() -> None:
  """Docstring."""
  adapter: SassAdapter = SassAdapter()
  code: str = """
    L_2:
    IADD3 R0, R0, 1
    FFMA R0, R1, R2, R3
    BRA L_2
    """
  adapter.parse_sass_to_graph(code)


def test_sass_loop_print() -> None:
  """Docstring."""
  adapter: SassAdapter = SassAdapter()
  code: str = """
    // comment
    /* block comment */
    L_1:
    MOV R0, R1
    L_2:
    IADD3 R0, R0, 1
    BRA L_2
    """
  g: LogicalGraph = adapter.parse_sass_to_graph(code)
  print(g.nodes)
