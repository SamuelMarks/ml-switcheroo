"""Test suite for the Sass module."""

import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_sass_adapter_init() -> None:
  """Verifies the behavior of SASS adapter initialization."""
  adapter = SassAdapter()
  assert adapter.display_name == "NVIDIA SASS"
  assert adapter.ui_priority == 150
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_sass_properties() -> None:
  """Verifies the behavior of SASS properties."""
  adapter = SassAdapter()
  assert adapter.import_alias == ("sass", "asm")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base is None
  config: dict[str, typing.Any] = adapter.test_config
  assert "// SASS Header" in config["import"]
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
  assert "FADD" in examples["tier1_math"]


def test_sass_missing_coverage() -> None:
  """Docstring."""
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
  code_loop: str = """
  // comment
  L_START:
  FADD R1, R1, R2
  ISETP.LT.AND P0, PT, R5, 128, PT;
  FFMA R1, R3, R5, R1
  BRA L_START;
  L_LABEL:
  """
  graph_loop: typing.Any = adapter.parse_sass_to_graph(code_loop)
  nodes_loop: list[typing.Any] = list(graph_loop.nodes.values())
  # The block "entry" branches to "L_START", "L_START" branches to "L_START".
  # "L_START" is a loop block containing FFMA.
  assert any(n.op_type == "Conv2d" for n in nodes_loop)

  code_no_loop: str = """
  FFMA R1, R3, R5, R1
  """
  graph_no_loop: typing.Any = adapter.parse_sass_to_graph(code_no_loop)
  nodes_no_loop: list[typing.Any] = list(graph_no_loop.nodes.values())
  assert len(nodes_no_loop) == 1
  assert nodes_no_loop[0].op_type == "Linear"


# --- Merged from test_sass_extra.py ---


def test_sass_missing_methods() -> None:
  """Docstring."""
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

  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)

  ex: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier1_math" in ex
  pass

  adapter.apply_wiring({})
  assert adapter.get_doc_url("api") is None


def test_sass_parse_sass_to_graph() -> None:
  """Docstring."""
  adapter = SassAdapter()

  # Test empty graph
  empty_graph: typing.Any = adapter.parse_sass_to_graph("// just a comment")
  assert len(empty_graph.nodes) == 0

  # Code with a loop
  # entry block (implicit) falls through to L_LOOP
  # L_LOOP has no FFMA, so it gets LoopControl
  # L_BODY has FFMA and branches back to L_LOOP
  code: str = """
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

  graph1: typing.Any = adapter.parse_sass_to_graph(code)
  assert graph1.name == "Model"
  nodes1: list[typing.Any] = list(graph1.nodes.values())
  # L_LOOP is in loop, no FFMA -> LoopControl
  # L_BODY is in loop, has FFMA -> Conv2d
  # entry is not in loop, no FFMA -> no node added
  assert len(nodes1) == 2
  op_types: set[str] = {n.op_type for n in nodes1}
  assert "LoopControl" in op_types
  assert "Conv2d" in op_types

  # code without loop
  code_no_loop: str = """
    FFMA R1, R2, R3, R4
    """
  graph2: typing.Any = adapter.parse_sass_to_graph(code_no_loop)
  nodes2: list[typing.Any] = list(graph2.nodes.values())
  assert len(nodes2) == 1
  assert nodes2[0].op_type == "Linear"
