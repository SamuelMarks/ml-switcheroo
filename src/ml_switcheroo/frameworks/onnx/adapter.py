"""ONNX Framework Adapter."""

import logging
import textwrap
from typing import List, Tuple, Dict, Any, Optional, Set

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    onnx = None
    helper = None
    TensorProto = None

from ml_switcheroo.frameworks.base import (
    register_framework, StructuralTraits, PluginTraits,
    StandardMap, ImportConfig, StandardCategory, InitMode, GhostRef
)
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.loader import load_definitions

@register_framework("onnx")
class OnnxFramework:
    """Adapter for ONNX Graph representation.
    
    Generates syntax that utilizes `onnx.helper` to build Protobuf models.
    """
    display_name: str = "ONNX"
    ui_priority: int = 5
    inherits_from: Optional[str] = None
    
    def __init__(self) -> None:
        """Initialize ONNX framework adapter."""
        self._mode = InitMode.LIVE if onnx is not None else InitMode.GHOST
        self._snapshot_data: Dict[str, Any] = {}
        
    @property
    def search_modules(self) -> List[str]:
        """Modules to search for ONNX APIs."""
        return ["onnx.helper", "onnx.numpy_helper"] if self._mode == InitMode.LIVE else []
        
    @property
    def unsafe_submodules(self) -> Set[str]:
        """Submodules to avoid scanning."""
        return {"onnx.reference", "onnx.backend"}
        
    @property
    def import_alias(self) -> Tuple[str, str]:
        """Canonical import alias for ONNX."""
        return ("onnx", "onnx")
        
    @property
    def import_namespaces(self) -> Dict[str, ImportConfig]:
        """Namespace configurations."""
        return {
            "onnx": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="onnx"),
            "onnx.helper": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="helper"),
        }
        
    @property
    def discovery_heuristics(self) -> Dict[str, List[str]]:
        """Heuristics to identify ONNX APIs."""
        return {"extras": [r"onnx\.helper\.", r"helper\."]}
        
    @property
    def test_config(self) -> Dict[str, str]:
        """Configuration for generated tests."""
        return {
            "import": "import onnx\nfrom onnx import helper, TensorProto\nimport numpy as np",
            "convert_input": "helper.make_tensor_value_info('{np_var}', TensorProto.FLOAT, np.shape({np_var}))",
            "to_numpy": "{res_var}",
            "jit_template": "{fn}",
        }
        
    @property
    def harness_imports(self) -> List[str]:
        """Imports for the verification harness."""
        return ["import onnx", "from onnx import helper"]
        
    def get_harness_init_code(self) -> str:
        """Initialization code for the harness."""
        return ""
        
    @property
    def declared_magic_args(self) -> List[str]:
        """Magic arguments used by the framework."""
        return []
        
    @property
    def structural_traits(self) -> StructuralTraits:
        """Structural traits describing ONNX logic."""
        return StructuralTraits(
            module_base=None,
            forward_method="make_model",
            inject_magic_args=[],
            requires_super_init=False,
            lifecycle_strip_methods=[],
            lifecycle_warn_methods=[],
            jit_static_args=[]
        )
        
    @property
    def plugin_traits(self) -> PluginTraits:
        """Plugin traits."""
        return PluginTraits(
            has_numpy_compatible_arrays=False,
            requires_explicit_rng=False,
            requires_functional_control_flow=True,
            enforce_purity_analysis=False,
            strict_materialization_method=None
        )
        
    @property
    def rng_seed_methods(self) -> List[str]:
        """Seed methods."""
        return []
        
    @property
    def definitions(self) -> Dict[str, StandardMap]:
        """Semantic definitions loaded from onnx.json."""
        return load_definitions("onnx")
        
    @property
    def specifications(self) -> Dict[str, Any]:
        """Operation specifications."""
        return {}
        
    def collect_api(self, category: StandardCategory) -> List[GhostRef]:
        """Collects APIs for a category."""
        return []
        
    def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
        """Applies wiring to a snapshot."""
        pass
        
    def convert(self, data: Any) -> Any:
        """Converts data to an ONNX compatible format."""
        return data
        
    def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
        """ONNX models are device agnostic in definition."""
        return "None"
        
    def get_device_check_syntax(self) -> str:
        """Checks for device availability."""
        return "True"
        
    def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
        """Splits an RNG key."""
        return ""
        
    def get_serialization_imports(self) -> List[str]:
        """Imports for serialization."""
        return ["import onnx"]
        
    def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
        """Syntax for serializing/deserializing."""
        if op == "save":
            return f"onnx.save({object_arg}, {file_arg})"
        elif op == "load":
            return f"onnx.load({file_arg})"
        return ""
        
    def get_weight_conversion_imports(self) -> List[str]:
        """Imports for weight conversion scripts."""
        return ["import onnx", "from onnx import numpy_helper"]
        
    def get_weight_load_code(self, path_var: str) -> str:
        """Code to load weights into raw_state dict."""
        return textwrap.dedent(f"""
            model = onnx.load({path_var})
            raw_state = {{init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}}
        """)
        
    def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
        """Expression to convert a tensor to numpy."""
        return f"onnx.numpy_helper.to_array({tensor_var})"
        
    def get_weight_save_code(self, state_var: str, path_var: str) -> str:
        """Code to save a raw_state dict to a model file."""
        return textwrap.dedent(f"""
            # Requires an existing model to append initializers
            initializers = [onnx.numpy_helper.from_array(v, name=k) for k, v in {state_var}.items()]
            graph = onnx.helper.make_graph([], 'weights_only', [], [], initializer=initializers)
            model = onnx.helper.make_model(graph)
            onnx.save(model, {path_var})
        """)
        
    def get_doc_url(self, api_name: str) -> Optional[str]:
        """URL to documentation."""
        return f"https://onnx.ai/onnx/operators/onnx__{api_name}.html"
        
    def get_tiered_examples(self) -> Dict[str, str]:
        """Examples for the demo UI."""
        return {
            "tier1_math": textwrap.dedent("""\
                import onnx
                from onnx import helper, TensorProto
                
                # Define simple Add graph
                X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
                Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
                Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 2])
                
                node = helper.make_node('Add', inputs=['X', 'Y'], outputs=['Z'])
                
                graph = helper.make_graph([node], 'test-model', [X], [Z])
                model = helper.make_model(graph, producer_name='ml-switcheroo')
            """)
        }
        
    def get_to_numpy_code(self) -> str:
        """Code to extract a numpy array from an object."""
        return "if hasattr(obj, 'numpy'): return obj.numpy()"
