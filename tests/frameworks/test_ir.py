"""Unit tests for the Intermediate Representation (IR) Framework Adapter."""

import json
from typing import Any, Dict

from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.frameworks.base import ImportConfig
from ml_switcheroo.frameworks.ir import IrAdapter


def test_ir_adapter_registration() -> None:
  """Verify that the IR adapter is properly registered under both aliases."""
  frameworks = available_frameworks()
  assert "ir" in frameworks
  assert "ml_switcheroo_ir" in frameworks

  adapter_ir = get_adapter("ir")
  adapter_sw = get_adapter("ml_switcheroo_ir")
  assert isinstance(adapter_ir, IrAdapter)
  assert isinstance(adapter_sw, IrAdapter)


def test_ir_adapter_basic_properties() -> None:
  """Verify metadata properties on the IrAdapter."""
  adapter = IrAdapter()
  assert adapter.display_name == "ML-Switcheroo IR (Intermediate Representation)"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 85
  assert adapter._mode.value == "live"
  assert adapter._snapshot_data == {}


def test_ir_adapter_import_configs() -> None:
  """Verify import alias and namespace configurations."""
  adapter = IrAdapter()
  pkg, alias = adapter.import_alias
  assert pkg == "ml_switcheroo_ir"
  assert alias == "sw_ir"

  namespaces = adapter.import_namespaces
  assert "ml_switcheroo_ir" in namespaces
  cfg1 = namespaces["ml_switcheroo_ir"]
  assert isinstance(cfg1, ImportConfig)
  assert cfg1.tier == SemanticTier.ARRAY_API
  assert cfg1.recommended_alias == "sw_ir"

  assert "ml_switcheroo_ir.schema" in namespaces
  cfg2 = namespaces["ml_switcheroo_ir.schema"]
  assert isinstance(cfg2, ImportConfig)
  assert cfg2.tier == SemanticTier.NEURAL
  assert cfg2.recommended_alias == "sw_schema"

  assert "ml_switcheroo_ir.types" in namespaces
  cfg3 = namespaces["ml_switcheroo_ir.types"]
  assert isinstance(cfg3, ImportConfig)
  assert cfg3.tier == SemanticTier.EXTRAS
  assert cfg3.recommended_alias == "sw_types"


def test_ir_adapter_test_config_and_harness() -> None:
  """Verify test harness configurations and code generation."""
  adapter = IrAdapter()
  config = adapter.test_config
  assert "import" in config
  assert "convert_input" in config
  assert "to_numpy" in config

  imports = adapter.harness_imports
  assert any("ml_switcheroo_ir" in imp for imp in imports)

  init_code = adapter.get_harness_init_code()
  assert "Validator" in init_code

  to_numpy = adapter.get_to_numpy_code()
  assert "np.asarray" in to_numpy


def test_ir_adapter_traits_and_specs() -> None:
  """Verify traits, supported tiers, and operation specifications."""
  adapter = IrAdapter()
  tiers = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers
  assert SemanticTier.NEURAL in tiers
  assert SemanticTier.EXTRAS in tiers

  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  assert adapter.specifications == {}

  structural = adapter.structural_traits
  assert structural.module_base == "ml_switcheroo_ir.LogicalGraph"
  assert structural.forward_method == "build"
  assert not structural.requires_super_init

  plugin = adapter.plugin_traits
  assert plugin.model_extra is not None or hasattr(plugin, "supports_dynamic_shapes")


def test_ir_adapter_device_and_rng() -> None:
  """Verify device and RNG syntax generation."""
  adapter = IrAdapter()
  syntax_no_idx = adapter.get_device_syntax("cuda")
  assert syntax_no_idx == "mesh_axis='cuda'"

  syntax_with_idx = adapter.get_device_syntax("cuda", "0")
  assert syntax_with_idx == "mesh_axis='cuda:0'"

  assert adapter.get_device_check_syntax() == "True"

  rng_split = adapter.get_rng_split_syntax("rng_in", "key_out")
  assert "key_out = sw_ir.split_prng(rng_in)" == rng_split


def test_ir_adapter_serialization_and_weights() -> None:
  """Verify model and weight serialization hooks."""
  adapter = IrAdapter()
  assert "import json" in adapter.get_serialization_imports()

  save_syntax = adapter.get_serialization_syntax("save", "'model.json'", "graph")
  assert "write_text" in save_syntax

  load_syntax = adapter.get_serialization_syntax("load", "'model.json'")
  assert "from_json" in load_syntax

  assert adapter.get_weight_conversion_imports() == []
  assert "loading" in adapter.get_weight_load_code("path.pt").lower()
  assert adapter.get_tensor_to_numpy_expr("t_var") == "np.asarray(t_var)"
  assert "saving" in adapter.get_weight_save_code("weights", "out.pt").lower()


def test_ir_adapter_wiring_and_doc_url() -> None:
  """Verify wiring hook and documentation URL lookup."""
  adapter = IrAdapter()
  adapter.apply_wiring({})
  doc_url = adapter.get_doc_url("LogicalGraph")
  assert doc_url is not None
  assert "LogicalGraph.md" in doc_url


def test_ir_adapter_convert() -> None:
  """Verify convert method with valid JSON, invalid string, and raw objects."""
  adapter = IrAdapter()

  graph_dict: Dict[str, Any] = {
    "name": "LinearNet",
    "nodes": {
      "in1": {"id": "in1", "op_type": "Input"},
      "out1": {"id": "out1", "op_type": "Output", "inputs": ["in1"]},
    },
    "outputs": ["out1"],
  }
  json_str = json.dumps(graph_dict)

  converted = adapter.convert(json_str)
  assert hasattr(converted, "name")
  assert getattr(converted, "name") == "LinearNet"

  invalid_str = "not a valid json {{"
  assert adapter.convert(invalid_str) == invalid_str

  raw_dict = {"key": 123}
  assert adapter.convert(raw_dict) == raw_dict


def test_ir_adapter_examples() -> None:
  """Verify example code generation."""
  adapter = IrAdapter()
  example = adapter.get_example_code()
  assert "SampleModel" in example
  assert "LogicalGraph" in example

  tiered = adapter.get_tiered_examples()
  assert "tier1_math" in tiered
  assert "tier2_neural" in tiered
  assert "tier3_extras" in tiered
  assert tiered["tier1_math"] == example


def test_ir_adapter_definitions() -> None:
  """Verify loading operation definitions for IR."""
  adapter = IrAdapter()
  defs = adapter.definitions
  assert isinstance(defs, dict)
