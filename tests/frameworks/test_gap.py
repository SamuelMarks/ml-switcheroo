import pytest
from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.enums import SemanticTier


def test_mlir_adapter_full():
  adapter = MlirAdapter()
  assert adapter.search_modules == []
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("mlir", "sw")
  assert adapter.import_namespaces == {}
  assert adapter.discovery_heuristics == {}

  assert "import" in adapter.test_config
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None

  defs = adapter.definitions
  assert isinstance(defs, dict)
  assert adapter.specifications == {}
  assert adapter.rng_seed_methods == []

  # Needs a mock category but any string might fail if typed, pass None or fake
  assert adapter.collect_api("math") == []
  assert adapter.get_device_syntax("cpu") == "// Target: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == "// Split RNG: rng -> key"
  assert adapter.get_serialization_imports() == []

  assert adapter.get_serialization_syntax("save", "f", "obj") == "// Save obj to f"
  assert adapter.get_serialization_syntax("load", "f") == "// Load from f"
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights loading not supported in MLIR adapter"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("s", "p") == "# Weights saving not supported in MLIR adapter"

  adapter.apply_wiring({})
  assert adapter.get_doc_url("any") is None
  assert adapter.convert(123) == "123"
  assert "Example MLIR" in adapter.get_example_code()

  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples


def test_rdna_adapter_full():
  adapter = RdnaAdapter()
  assert adapter.search_modules == []
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("rdna", "asm")
  assert adapter.import_namespaces == {}
  assert adapter.discovery_heuristics == {}

  assert "import" in adapter.test_config
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None

  defs = adapter.definitions
  assert isinstance(defs, dict)
  assert adapter.specifications == {}
  assert adapter.rng_seed_methods == []

  assert adapter.collect_api("math") == []
  assert adapter.get_device_syntax("cpu") == "; Target Device: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []

  assert adapter.get_serialization_syntax("save", "f", "obj") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "; Weights loading not supported in RDNA adapter"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("s", "p") == "; Weights saving not supported in RDNA adapter"

  adapter.apply_wiring({})
  assert adapter.get_doc_url("any") == "https://gpuopen.com/learn/rdna-performance-guide/?q=any"
  assert adapter.convert(123) == "123"

  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples


def test_sass_adapter_full():
  adapter = SassAdapter()
  assert adapter.search_modules == []
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("sass", "asm")
  assert adapter.import_namespaces == {}
  assert adapter.discovery_heuristics == {}

  assert "import" in adapter.test_config
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None

  defs = adapter.definitions
  assert isinstance(defs, dict)
  assert adapter.specifications == {}
  assert adapter.rng_seed_methods == []

  assert adapter.collect_api("math") == []
  assert adapter.get_device_syntax("cpu") == "// Target Device: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []

  assert adapter.get_serialization_syntax("save", "f", "obj") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "// Weights loading not supported in SASS adapter"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("s", "p") == "// Weights saving not supported in SASS adapter"

  adapter.apply_wiring({})
  assert adapter.get_doc_url("any") is None
  assert adapter.convert(123) == "123"

  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
