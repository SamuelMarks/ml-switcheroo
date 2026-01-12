#!/bin/sh
set -e

# 1. Clean up the rewriter package
cat > src/ml_switcheroo/core/rewriter/__init__.py << 'EOF'
"""
Rewriter Package.

Exposes the core transformation pipeline components.
"""

from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.core.rewriter.passes.api import ApiPass, ApiTransformer
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer

__all__ = [
    "RewriterContext",
    "RewriterPipeline",
    "RewriterPass",
    "StructuralPass",
    "StructuralTransformer",
    "ApiPass",
    "ApiTransformer",
    "AuxiliaryPass",
    "AuxiliaryTransformer",
]
EOF

# 2. Update conftest.py with TestRewriter
cat > tests/conftest.py << 'EOF'
import sys
import pytest
import warnings
from pathlib import Path
from typing import Callable, Optional

# --- FIX: Global Warning Suppression for Collection Phase ---
# Suppress the Keras/NumPy 'np.object' FutureWarning that crashes collection
warnings.filterwarnings("ignore", message=".*np\\.object.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow.*")
# ------------------------------------------------------------

# Add src to path so we can import 'ml_switcheroo' without installing it
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Force load of default adapters so they provide the "clean state" baseline
try:
    import ml_switcheroo.frameworks
    from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY
except ImportError:
    _ADAPTER_REGISTRY = {}

# Import Rewriter components for TestRewriter shim
from ml_switcheroo.core.rewriter import (
    RewriterContext,
    RewriterPipeline,
    StructuralPass,
    ApiPass,
    AuxiliaryPass
)

class TestRewriter:
    """
    Test-scoped shim replacing the legacy PivotRewriter.
    Wraps the RewriterPipeline to allow tests to execute transformations
    deterministically without the full Engine overhead.
    """
    def __init__(self, semantics, config, symbol_table=None):
        self.context = RewriterContext(semantics, config, symbol_table)
        self.pipeline = RewriterPipeline([
            StructuralPass(),
            ApiPass(),
            AuxiliaryPass()
        ])

    @property
    def ctx(self):
        """Access hook context for assertions."""
        return self.context.hook_context

    @property
    def semantics(self):
        return self.context.semantics

    def convert(self, tree):
        """Runs the pipeline on a CST module."""
        return self.pipeline.run(tree, self.context)

class SnapshotAssert:
    """
    Simple snapshot comparison logic to verify CLI output stability.
    """

    def __init__(self, request: pytest.FixtureRequest):
        self.request = request
        self.test_name = request.node.name
        self.module_path = Path(request.node.fspath).parent
        self.snapshot_dir = self.module_path / "__snapshots__"
        self.update_mode = request.config.getoption("--update-snapshots", default=False)

    def assert_match(self, content: str, extension: str = "txt", normalizer: Optional[Callable[[str], str]] = None):
        """
        Compares content against stored file.

        Args:
            content: The actual output string.
            extension: File extension (default 'txt', 'json', etc).
            normalizer: Optional function to clean both content and expected string before comparison.
        """
        if not self.snapshot_dir.exists():
            self.snapshot_dir.mkdir(parents=True)

        snapshot_file = self.snapshot_dir / f"{self.test_name}.{extension}"

        # Normalize line endings
        content = content.replace("\r\n", "\n")

        if self.update_mode or not snapshot_file.exists():
            # We write the raw content (without external normalization) to disk
            normalized_to_write = normalizer(content) if normalizer else content
            snapshot_file.write_text(normalized_to_write, encoding="utf-8")
            if self.update_mode:
                return

        expected = snapshot_file.read_text(encoding="utf-8").replace("\r\n", "\n")

        lhs = content
        rhs = expected

        if normalizer:
            lhs = normalizer(lhs)
            rhs = normalizer(rhs)

        # Simple assertion. Diff tools in IDE handles failures well.
        assert lhs == rhs, (
            f"Snapshot mismatch for {snapshot_file.name}. Run check script or pytest with --update-snapshots to accept changes."
        )

@pytest.fixture
def snapshot(request):
    """Fixture to assert text matches a stored snapshot."""
    return SnapshotAssert(request)

@pytest.fixture(autouse=True)
def isolate_framework_registry():
    """
    Ensures that modifications to the framework adapter registry
    (adding custom frameworks for tests) do not leak between tests.
    """
    # Capture the state (which ideally contains torch, jax, etc. loaded via imports above)
    original_registry = _ADAPTER_REGISTRY.copy()
    yield
    # Restore state after test completion
    _ADAPTER_REGISTRY.clear()
    _ADAPTER_REGISTRY.update(original_registry)

def pytest_addoption(parser):
    """Add CLI flag to update snapshots."""
    parser.addoption("--update-snapshots", action="store_true", default=False, help="Update snapshots for visual tests")
EOF

# 3. Update functionality tests

cat > tests/functionality/test_rewriter.py << 'EOF'
"""
Comprehensive Integration Tests for the TestRewriter pipeline.

Verifies:
1.  **API Swapping**: Calls are correctly mapped (torch.abs -> jax.numpy.abs).
2.  **Arg Normalization**: Params are pivoted via the abstract spec (input -> x -> a).
3.  **Recursive Rewriting**: Nested calls (`abs(neg(x))`) are transformed inside-out.
4.  **Complex Statements**: Calls inside Return, Assign, and List structures work.
5.  **Alias Resolution**: Integration with alias map works end-to-end.
6.  **Pass-through**: Unknown APIs are preserved.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig

class MockSemantics(SemanticsManager):
    """
    Mock Manager that skips file I/O and provides deterministic test data.
    """

    def __init__(self):
        # Skip super() init to avoid loading real files
        self.data = {}
        self.import_data = {}
        self._reverse_index = {}
        self._key_origins = {}
        self.framework_configs = {}

        # 1. Simple Swap: abs
        # torch.abs(x) -> jax.numpy.abs(x)
        self._inject("abs", ["x"], "torch.abs", "jax.numpy.abs")

        # 2. Argument Rename: sum
        # Torch: sum(input) -> Std: sum(x) -> Jax: sum(a)
        self._inject("sum", ["x"], "torch.sum", "jax.numpy.sum", s_args={"x": "input"}, t_args={"x": "a"})

        # 3. Unary Op: neg
        self._inject("neg", ["x"], "torch.neg", "jax.numpy.negative")

        # 4. Binary Op: add
        self._inject("add", ["x", "y"], "torch.add", "jax.numpy.add")

    def get_framework_config(self, framework: str):
        return self.framework_configs.get(framework, {})

    def _inject(self, name, std_args, s_api, t_api, s_args=None, t_args=None):
        s_def = {"api": s_api}
        if s_args:
            s_def["args"] = s_args

        t_def = {"api": t_api}
        if t_args:
            t_def["args"] = t_args

        self.data[name] = {"std_args": std_args, "variants": {"torch": s_def, "jax": t_def}}
        self._reverse_index[s_api] = (name, self.data[name])

@pytest.fixture
def rewriter():
    semantics = MockSemantics()
    config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
    return TestRewriter(semantics, config)

def rewrite(rewriter, code):
    """Helper to parse and return code string."""
    tree = cst.parse_module(code)
    # Use convert() which wraps pipeline.run()
    new_tree = rewriter.convert(tree)
    return new_tree.code

def test_simple_api_swap(rewriter):
    """
    Input:  y = torch.abs(x)
    Output: y = jax.numpy.abs(x)
    """
    code = "y = torch.abs(x)"
    result = rewrite(rewriter, code)
    assert "jax.numpy.abs(x)" in result

def test_argument_renaming(rewriter):
    """
    Input:  y = torch.sum(input=t)
    Output: y = jax.numpy.sum(a=t)
    """
    code = "y = torch.sum(input=t)"
    result = rewrite(rewriter, code)
    assert "jax.numpy.sum(a=t)" in result

def test_nested_calls_recursive(rewriter):
    """
    Input:  y = torch.abs(torch.neg(x))
    Output: y = jax.numpy.abs(jax.numpy.negative(x))
    """
    code = "y = torch.abs(torch.neg(x))"
    result = rewrite(rewriter, code)

    assert "jax.numpy.abs" in result
    assert "jax.numpy.negative(x)" in result
    assert "torch" not in result

def test_complex_nested_structure(rewriter):
    """
    Input:  y = torch.add(torch.abs(a), torch.neg(b))
    Output is fully converted.
    """
    code = "y = torch.add(torch.abs(a), torch.neg(b))"
    result = rewrite(rewriter, code)

    assert "jax.numpy.add" in result
    assert "jax.numpy.abs(a)" in result
    assert "jax.numpy.negative(b)" in result

def test_return_statement_rewrite(rewriter):
    """
    Verify rewrites work inside return statements.
    Input:  return torch.abs(x)
    Output: return jax.numpy.abs(x)
    """
    code = "def f(x):\n    return torch.abs(x)"
    result = rewrite(rewriter, code)
    assert "return jax.numpy.abs(x)" in result

def test_function_arg_rewrite(rewriter):
    """
    Verify rewrites work when call is an argument to another function.
    Input:  print(torch.abs(x))
    Output: print(jax.numpy.abs(x))
    """
    code = "print(torch.abs(x))"
    result = rewrite(rewriter, code)
    assert "jax.numpy.abs(x)" in result

def test_list_element_rewrite(rewriter):
    """
    Input:  l = [torch.abs(x), torch.neg(y)]
    Output: l = [jax.numpy.abs(x), jax.numpy.negative(y)]
    """
    code = "l = [torch.abs(x), torch.neg(y)]"
    result = rewrite(rewriter, code)
    assert "jax.numpy.abs(x)" in result
    assert "jax.numpy.negative(y)" in result

def test_dict_value_rewrite(rewriter):
    """
    Input:  d = {'val': torch.abs(x)}
    Output: d = {'val': jax.numpy.abs(x)}
    """
    code = "d = {'val': torch.abs(x)}"
    result = rewrite(rewriter, code)
    assert "{'val': jax.numpy.abs(x)}" in result

def test_pass_through_unknown(rewriter):
    """
    Verify unknown APIs are preserved verbatim.
    """
    code = "y = torch.unknown_func(x)"
    result = rewrite(rewriter, code)
    assert "torch.unknown_func(x)" in result

def test_aliased_usage(rewriter):
    """
    Verify that local aliases defined in import override source rules.
    Input:
        import torch as t
        y = t.abs(x)
    Output:
        ...
        y = jax.numpy.abs(x)
    """
    code = """
import torch as t
y = t.abs(x)
"""
    result = rewrite(rewriter, code)
    assert "jax.numpy.abs(x)" in result
EOF

cat > tests/functionality/test_rewriter_arg_normalization.py << 'EOF'
"""
Tests for Argument Normalization Logic in TestRewriter.

Verifies that the Rewriter correctly pivots argument names and positions
based on the abstract specification.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig

class MockArgSemantics(SemanticsManager):
    """
    Mock Data for Argument Normalization tests.
    """

    def __init__(self) -> None:
        """Initializes with specific argument mapping scenarios."""
        self.data = {}
        self.import_data = {}
        self._reverse_index = {}
        self._key_origins = {}

        # NEW: Populate framework configs to support dynamic module detection
        self.framework_configs = {
            "torch": {"alias": {"module": "torch", "name": "torch"}},
            "jax": {"alias": {"module": "jax.numpy", "name": "jnp"}},
            "experimental_fw": {"alias": {"module": "exp.net", "name": "exp"}},
        }

        # 1. Complex Renaming ('sum')
        # Source: sum(input, dim) -> Standard: sum(x, axis) -> Target: sum(a, axis)
        self._inject_op(
            op_name="sum",
            std_args=["x", "axis"],
            variants={
                "torch": {
                    "api": "torch.sum",
                    "args": {"x": "input", "axis": "dim"},
                },
                "jax": {
                    "api": "jax.numpy.sum",
                    "args": {"x": "a", "axis": "axis"},
                },
            },
        )

        # 2. Simple Positional ('div')
        self._inject_op(
            op_name="div",
            std_args=["x", "y"],
            variants={
                "torch": {"api": "torch.div"},
                "jax": {"api": "jax.numpy.divide"},
            },
        )

        # 3. Typed Arguments Specification ('randint')
        # Standard: randint(low: int, high: int)
        self._inject_op(
            op_name="randint",
            std_args=[("low", "int"), ("high", "int")],
            variants={
                "torch": {"api": "torch.randint"},
                "jax": {"api": "jax.random.randint"},
            },
        )

        # 4. Injection Scenario ('normalize')
        self._inject_op(
            op_name="normalize",
            std_args=["x"],
            variants={
                "torch": {"api": "torch.normalize"},
                "jax": {"api": "jax.nn.normalize", "inject_args": {"epsilon": 1e-5, "flag": True}},
            },
        )

        # 5. Value Mapping Scenario ('reduce')
        self._inject_op(
            op_name="reduce",
            std_args=["x", "val"],
            variants={
                "torch": {"api": "torch.reduce", "args": {"val": "reduction"}},
                "jax": {"api": "jax.reduce", "args": {"val": "mode"}, "arg_values": {"val": {"mean": "'avg'", "0": "'none'"}}},
            },
        )

        # 6. Method Style Test ('method_op')
        self._inject_op(
            op_name="method_op",
            std_args=["x", "y"],
            variants={
                "torch": {"api": "torch.method_op"},
                "jax": {"api": "jax.method_op"},
            },
        )

    def _inject_op(self, op_name, std_args, variants):
        self.data[op_name] = {"std_args": std_args, "variants": variants}
        for _, details in variants.items():
            if "api" in details:
                self._reverse_index[details["api"]] = (op_name, self.data[op_name])

@pytest.fixture
def engine() -> TestRewriter:
    """Returns a Rewriter for Torch -> JAX arg testing."""
    semantics = MockArgSemantics()
    config = RuntimeConfig(source_framework="torch", target_framework="jax")
    return TestRewriter(semantics, config)

def rewrite_code(rewriter: TestRewriter, code: str) -> str:
    """Parses code, applies rewriter, and returns generated code."""
    tree = cst.parse_module(code)
    return rewriter.convert(tree).code

def test_keyword_translation(engine: TestRewriter) -> None:
    """
    Input:  `torch.sum(input=z, dim=1)`
    Logic:  input -> x -> a, dim -> axis -> axis.
    Output: `jax.numpy.sum(a=z, axis=1)`
    """
    code = "res = torch.sum(input=temp, dim=1)"
    result = rewrite_code(engine, code)
    assert "res = jax.numpy.sum(a=temp, axis=1)" in result

def test_positional_passthrough(engine: TestRewriter) -> None:
    code = "res = torch.div(val_1, val_2)"
    result = rewrite_code(engine, code)
    assert "res = jax.numpy.divide(val_1, val_2)" in result

def test_mixed_args_normalization(engine: TestRewriter) -> None:
    code = "res = torch.sum(my_tensor, dim=2)"
    result = rewrite_code(engine, code)
    assert "res = jax.numpy.sum(my_tensor, axis=2)" in result

def test_unknown_keyword_passthrough(engine: TestRewriter) -> None:
    code = "res = torch.sum(x, keepdims=True)"
    result = rewrite_code(engine, code)
    assert "res = jax.numpy.sum(x, keepdims=True)" in result

def test_typed_arguments_handling(engine: TestRewriter) -> None:
    code = "r = torch.randint(low=0, high=10)"
    result = rewrite_code(engine, code)
    assert "r = jax.random.randint(low=0, high=10)" in result

def test_argument_injection(engine: TestRewriter) -> None:
    code = "y = torch.normalize(data)"
    result = rewrite_code(engine, code)

    assert "jax.nn.normalize(data" in result
    assert "epsilon=1e-05" in result
    assert "flag=True" in result

def test_argument_value_mapping_strings(engine: TestRewriter) -> None:
    code = "y = torch.reduce(x, reduction='mean')"
    result = rewrite_code(engine, code)

    assert "jax.reduce" in result
    assert "mode='avg'" in result

def test_module_alias_detection(engine: TestRewriter) -> None:
    """
    Test dynamic module alias detection logic.
    """
    # Case 1: Framework call
    code_fw = "torch.method_op(y)"
    res_fw = rewrite_code(engine, code_fw)
    clean_fw = res_fw.replace(" ", "")
    assert "(torch," not in clean_fw
    assert "(y)" in clean_fw

    # Case 2: Instance call
    code_inst = "my_obj.method_op(y)"
    # Ensure reverse lookup
    op_def = engine.semantics.data["method_op"]
    engine.semantics._reverse_index["my_obj.method_op"] = ("method_op", op_def)

    res_inst = rewrite_code(engine, code_inst)
    clean_inst = res_inst.replace(" ", "")
    assert "(my_obj,y)" in clean_inst
EOF

cat > tests/functionality/test_rewriter_bubbling.py << 'EOF'
import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig

class MockSemantics(SemanticsManager):
    def __init__(self):
        self.data = {"bad": {"variants": {"jax": None}}, "good": {"variants": {"jax": {"api": "j.good"}}}}
        self.import_data = {}
        self.framework_configs = {}

    def get_definition(self, name):
        if "bad" in name:
            return "bad", self.data["bad"]
        if "good" in name:
            return "good", self.data["good"]
        return None

    def resolve_variant(self, aid, t):
        return self.data.get(aid, {}).get("variants", {}).get(t)

    def is_verified(self, _id):
        return True

@pytest.fixture
def rewriter():
    return TestRewriter(MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True))

def rewrite_stmt(rewriter, code):
    tree = cst.parse_module(code)
    return rewriter.convert(tree).code

def test_single_failure_bubbling(rewriter):
    res = rewrite_stmt(rewriter, "x = torch.bad(y)")
    assert EscapeHatch.START_MARKER in res
    assert "No mapping" in res

def test_nested_failure_bubbling(rewriter):
    res = rewrite_stmt(rewriter, "x = torch.good(torch.bad(y))")
    assert EscapeHatch.START_MARKER in res
    assert "No mapping" in res

def test_multiple_failures_deduplicated(rewriter):
    res = rewrite_stmt(rewriter, "l = [torch.bad(1), torch.bad(2)]")
    assert res.count("No mapping") == 1

def test_unknown_strict_mode(rewriter):
    res = rewrite_stmt(rewriter, "y = torch.unknown(x)")
    assert EscapeHatch.START_MARKER in res
    assert "API 'torch.unknown' not found" in res

def test_unknown_lax_mode():
    rw = TestRewriter(MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False))
    res = rewrite_stmt(rw, "y = torch.unknown(x)")
    assert EscapeHatch.START_MARKER not in res
EOF

cat > tests/functionality/test_rewriter_constants.py << 'EOF'
"""
Tests for Attribute/Constant Rewriting.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig

class MockSemantics(SemanticsManager):
    def __init__(self):
        # Skip init to avoid file load
        self.data = {}
        self._reverse_index = {}
        self._key_origins = {}
        self.framework_configs = {}

        # 1. Constant: float32 -> float32
        self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})

        # 2. Function: abs -> abs
        self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})

    def get_framework_config(self, framework: str):
        return self.framework_configs.get(framework, {})

    def _inject_const(self, name, mapping):
        # Constants have no std_args
        self.data[name] = {"variants": {}}
        for fw, api in mapping.items():
            self.data[name]["variants"][fw] = {"api": api}
            self._reverse_index[api] = (name, self.data[name])

    def _inject_func(self, name, mapping):
        # Functions have std_args
        self.data[name] = {"variants": {}, "std_args": ["x"]}
        for fw, api in mapping.items():
            self.data[name]["variants"][fw] = {"api": api}
            self._reverse_index[api] = (name, self.data[name])

@pytest.fixture
def rewriter():
    config = RuntimeConfig(source_framework="torch", target_framework="jax")
    return TestRewriter(MockSemantics(), config)

def rewrite(rewriter, code):
    tree = cst.parse_module(code)
    return rewriter.convert(tree).code

def test_constant_rewrite_assignment(rewriter):
    """
    Input:  dtype = torch.float32
    Expect: dtype = jax.numpy.float32
    """
    code = "x = torch.float32"
    res = rewrite(rewriter, code)
    assert "jax.numpy.float32" in res
    assert "torch.float32" not in res

def test_constant_rewrite_argument(rewriter):
    """
    Input:  init(dtype=torch.float32)
    Expect: init(dtype=jax.numpy.float32)
    """
    code = "y = init(dtype=torch.float32)"
    res = rewrite(rewriter, code)
    assert "jax.numpy.float32" in res

def test_function_call_rewrite(rewriter):
    """
    Input:  y = torch.abs(x)
    Expect: y = jax.numpy.abs(x)
    """
    code = "y = torch.abs(x)"
    res = rewrite(rewriter, code)
    assert "jax.numpy.abs(x)" in res
    assert "torch.abs" not in res
EOF

cat > tests/functionality/test_rewriter_decorators.py << 'EOF'
"""
Tests for Decorator Rewriting Logic.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OpType

class MockDecoratorSemantics(SemanticsManager):
    """
    Mock Manager for decorator scenarios.
    """

    def __init__(self):
        # Skip init to avoid file load
        self.data = {}
        self._reverse_index = {}
        self._key_origins = {}
        self.import_data = {}
        self.framework_configs = {}

        # 1. Rename: torch.jit.script -> jax.jit
        self._inject("jit", "torch.jit.script", "jax.jit")

        # 2. Remove: torch.inference_mode -> None (for JAX)
        self._inject("inference_mode", "torch.inference_mode", None)

        # 3. Call-style: torch.compile -> jax.jit
        self._inject("compile", "torch.compile", "jax.jit")

    def get_framework_config(self, framework: str):
        return self.framework_configs.get(framework, {})

    def _inject(self, name, s_api, t_api):
        variants = {"torch": {"api": s_api}}
        if t_api is None:
            variants["jax"] = None  # Explicit removal
        else:
            variants["jax"] = {"api": t_api}

        self.data[name] = {
            "op_type": OpType.DECORATOR,
            "variants": variants,
            "std_args": ["fn"],
        }
        self._reverse_index[s_api] = (name, self.data[name])

@pytest.fixture
def rewriter():
    semantics = MockDecoratorSemantics()
    config = RuntimeConfig(source_framework="torch", target_framework="jax")
    return TestRewriter(semantics, config)

def rewrite(rewriter, code):
    tree = cst.parse_module(code)
    try:
        new_tree = rewriter.convert(tree)
        return new_tree.code
    except Exception as e:
        pytest.fail(f"Rewriter failed: {e}")

def test_decorator_renaming(rewriter):
    code = """
@torch.jit.script
def func(x):
    return x
"""
    result = rewrite(rewriter, code)
    assert "@jax.jit" in result
    assert "@torch.jit.script" not in result

def test_decorator_removal(rewriter):
    code = """
@torch.inference_mode
def func(x):
    return x
"""
    result = rewrite(rewriter, code)
    assert "@torch.inference_mode" not in result
    assert "def func(x):" in result

def test_call_decorator_renaming(rewriter):
    code = """
@torch.compile(fullgraph=True)
def func(x):
    pass
"""
    result = rewrite(rewriter, code)
    assert "@jax.jit(fullgraph=True)" in result
    assert "torch.compile" not in result

def test_multiple_decorators_mixed(rewriter):
    code = """
@torch.jit.script
@torch.inference_mode
def f():
    pass
"""
    result = rewrite(rewriter, code)
    assert "@jax.jit" in result
    assert "@torch.inference_mode" not in result
    assert "def f():" in result
EOF

cat > tests/functionality/test_rewriter_defaults.py << 'EOF'
"""
Tests for ODL Default Argument Injection with Rich Types.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig

@pytest.fixture
def manager():
    """Mock Semantics Manager with ODL definitions containing rich defaults."""
    mgr = SemanticsManager()
    mgr.data = {}
    mgr._reverse_index = {}
    mgr._key_origins = {}
    mgr.framework_configs = {}
    if not hasattr(mgr, "import_data"):
        mgr.import_data = {}

    # 1. LayerNorm: float default (native float type in python)
    op = {
        "std_args": [{"name": "x"}, {"name": "eps", "type": "float", "default": 1e-5}],
        "variants": {
            "torch": {"api": "torch.nn.LayerNorm", "args": {"eps": "eps"}},
            "jax": {"api": "jax.nn.layer_norm", "args": {"eps": "epsilon"}},
        },
    }

    mgr.data["LayerNorm"] = op
    mgr._reverse_index["torch.nn.LayerNorm"] = ("LayerNorm", op)

    # 2. Dropout
    op_drop = {
        "std_args": [{"name": "x"}, {"name": "p", "type": "float", "default": 0.5}],
        "variants": {
            "torch": {"api": "torch.dropout"},
            "jax": {"api": "jax.random.bernoulli", "args": {"p": "p"}},
        },
    }
    mgr.data["Dropout"] = op_drop
    mgr._reverse_index["torch.dropout"] = ("Dropout", op_drop)

    # Mock aliasing
    mgr.framework_configs["torch"] = {"alias": {"module": "torch", "name": "t"}}
    mgr.framework_configs["jax"] = {}

    return mgr

@pytest.fixture
def rewriter(manager):
    config = RuntimeConfig(source_framework="torch", target_framework="jax")
    return TestRewriter(manager, config)

def rewrite(rewriter, code):
    return rewriter.convert(cst.parse_module(code)).code

def test_inject_default_float(rewriter):
    code = "import torch\ny = torch.nn.LayerNorm(x)"
    res = rewrite(rewriter, code)
    # Check for valid float repr
    assert "epsilon=1e-05" in res or "epsilon=0.00001" in res

def test_preserve_explicit_eps(rewriter):
    code = "import torch\ny = torch.nn.LayerNorm(x, eps=0.1)"
    res = rewrite(rewriter, code)
    assert "epsilon=0.1" in res
    assert "1e-5" not in res

def test_inject_default_dropout(rewriter):
    code = "import torch\ny = torch.dropout(x)"
    res = rewrite(rewriter, code)
    assert "p=0.5" in res
EOF

cat > tests/functionality/test_rewriter_functional_unwrap.py << 'EOF'
"""
Tests for Functional Unwrapping logic in TestRewriter.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig

class MockUnwrapSemantics(SemanticsManager):
    """
    Mock Manager for unwrapping tests.
    """

    def __init__(self):
        self.data = {}
        self._reverse_index = {}
        self._key_origins = {}
        self.import_data = {}
        self.framework_configs = {}  # No special config implies OOP default

@pytest.fixture
def rewriter():
    semantics = MockUnwrapSemantics()
    config = RuntimeConfig(
        source_framework="jax",  # Validated Source
        target_framework="jax",  # Targeting NNX (which is JAX but OOP)
        strict_mode=False,
    )
    return TestRewriter(semantics, config)

def rewrite_code(rewriter, code):
    tree = cst.parse_module(code)
    try:
        new_tree = rewriter.convert(tree)
        return new_tree.code
    except Exception as e:
        pytest.fail(f"Rewrite failed: {e}")

def test_unwrap_call_only(rewriter):
    """
    Input: `z = self.layer.apply(variables, x) + 1`
    Output: `z = self.layer(x) + 1`
    """
    code = "z = self.layer.apply(variables, x) + 1"
    result = rewrite_code(rewriter, code)

    assert "self.layer(x)" in result
    assert "apply" not in result
    assert "variables" not in result

def test_unwrap_assignment_tuple(rewriter):
    """
    Input: `y, updates = self.layer.apply(vars, x)`
    Output: `y = self.layer(x)`
    """
    code = "y, updates = self.layer.apply(vars, x)"
    result = rewrite_code(rewriter, code)

    # Must unwrap assignment target
    assert "y = self.layer(x)" in result
    assert "updates" not in result
EOF

cat > tests/functionality/test_rewriter_lifecycle.py << 'EOF'
"""
Tests for Model Lifecycle Translation, Version Enforcement, and Deprecation Warnings.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch

class MockSemantics(SemanticsManager):
    """Minimal semantics manager with Trait Support."""

    def __init__(self):
        self.data = {}
        self._reverse_index = {}
        self._key_origins = {}
        self.import_data = {}
        self._known_rng_methods = set()
        self._validation_status = {}  # Added to prevent attribute error in base

        # Add a basic op to ensure standard rewrites still work alongside stripping
        self._inject("abs", "torch.abs", "jax.numpy.abs")
        # Add basic types to prevent Attribute lookup failures in strict mode
        self._inject("float32", "torch.float32", "jax.numpy.float32")

        # Version constrained ops
        self._inject("future_op", "torch.future", "jax.future", min_v="9.0.0")
        self._inject("legacy_op", "torch.legacy", "jax.legacy", max_v="0.0.1")

        # Deprecated ops
        self._inject("old_scatter", "torch.old_scatter", "jax.scatter", deprecated=True, replaced_by="Scatter")
        self._inject("unsafe_op", "torch.unsafe", "jax.unsafe", deprecated=True)

        # --- FIX: Populate framework configs for SOURCE traits ---
        self.framework_configs = {
            "torch": {
                "traits": {
                    "lifecycle_strip_methods": ["to", "cpu", "cuda", "detach"],
                    "lifecycle_warn_methods": ["eval", "train"],
                }
            },
            "jax": {
                # Mock config with version
                "version": "1.0.0"
            },
        }

    def get_framework_config(self, framework: str):
        return self.framework_configs.get(framework, {})

    def is_verified(self, _id):
        return True

    def _inject(self, name, s_api, t_api, min_v=None, max_v=None, deprecated=False, replaced_by=None):
        tgt_var = {"api": t_api}
        if min_v:
            tgt_var["min_version"] = min_v
        if max_v:
            tgt_var["max_version"] = max_v

        self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": tgt_var}, "std_args": ["x"]}

        if deprecated:
            self.data[name]["deprecated"] = True
        if replaced_by:
            self.data[name]["replaced_by"] = replaced_by

        self._reverse_index[s_api] = (name, self.data[name])

@pytest.fixture
def rewriter():
    semantics = MockSemantics()
    config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
    return TestRewriter(semantics, config)

def rewrite(rewriter, code):
    """Executes the rewriter on the code string."""
    tree = cst.parse_module(code)
    try:
        new_tree = rewriter.convert(tree)
        return new_tree.code
    except Exception as e:
        pytest.fail(f"Rewriter crashed: {e}")

def test_strip_to_call(rewriter):
    """
    Input: x = tensor.to(device)
    Effect: .to() stripped.
    Output: x = tensor
            (Wrapped in warning markers)
    """
    code = "x = tensor.to(device)"
    result = rewrite(rewriter, code)

    # 1. Check Replacement: 'tensor.to(device)' -> 'tensor'
    assert "x = tensor" in result

    # 2. Check that .to is NOT present in the logic of the code
    is_to_present = any(".to(" in line and not line.strip().startswith("#") for line in result.splitlines())
    assert not is_to_present

    # 3. Check Warning Marker
    assert EscapeHatch.START_MARKER in result
    assert "Stripped framework-specific lifecycle method '.to()'" in result

def test_warn_on_eval_train(rewriter):
    """
    Input: model.eval()
    Effect: .eval() stripped (identity), warning attached.
    Output: model
    """
    code = "model.eval()"
    result = rewrite(rewriter, code)

    # This becomes an expression statement "model" (basically no-op)
    is_eval = any("model.eval" in line and not line.strip().startswith("#") for line in result.splitlines())
    assert not is_eval
    assert EscapeHatch.START_MARKER in result
    assert "Ignored model state method '.eval()'" in result

def test_version_constraint_check_min(rewriter):
    """
    Scenario: Op requires min_version="9.0.0". Target is "1.0.0".
    Expectation: Warning generated.
    """
    code = "y = torch.future(x)"
    result = rewrite(rewriter, code)

    assert "jax.future(x)" in result
    assert EscapeHatch.START_MARKER in result
    assert "Target jax@1.0.0 is older than required 9.0.0" in result

def test_deprecation_warning(rewriter):
    """
    Scenario: Op marked as deprecated.
    Expectation: Warning generated.
    """
    code = "y = torch.unsafe(x)"
    result = rewrite(rewriter, code)

    assert "jax.unsafe(x)" in result
    assert "Usage of deprecated operation 'unsafe_op'" in result
EOF

cat > tests/functionality/test_rewriter_state_mechanism.py << 'EOF'
"""
Tests for Generic State Mechanism handling.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import SemanticTier

class MockStateSemantics(SemanticsManager):
    """
    Mock Manager with arbitrary state configurations.
    """

    def __init__(self) -> None:
        self.data = {}
        self._reverse_index = {}
        self._key_origins = {}
        self.import_data = {}

        self.framework_configs = {
            "tensorflow": {"stateful_call": {"method": "apply", "prepend_arg": "variables"}},
            "mlx": {"stateful_call": {"method": "call_fn", "prepend_arg": "ctx"}},
        }

        # Define a 'Linear' operation that is considered stateful (Tier=Neural)
        self._inject(
            "Linear",
            SemanticTier.NEURAL,
            "torch",
            "torch.Linear",
            "tensorflow",
            "func.Dense",
        )

        # Add variant for MLX
        self.data["Linear"]["variants"]["mlx"] = {"api": "custom.Layer"}

    # --- FIX: Added method ---
    def get_framework_config(self, framework: str):
        return self.framework_configs.get(framework, {})

    def _inject(
        self,
        name: str,
        tier: SemanticTier,
        s_fw: str,
        s_api: str,
        t_fw: str,
        t_api: str,
    ) -> None:
        variants = {s_fw: {"api": s_api}, t_fw: {"api": t_api}}
        self.data[name] = {"variants": variants, "std_args": ["x"]}
        self._reverse_index[s_api] = (name, self.data[name])
        self._key_origins[name] = tier.value

@pytest.fixture
def rewriter() -> TestRewriter:
    """Rewriter defaulting to 'tensorflow' target (simulating functional)."""
    semantics = MockStateSemantics()
    config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
    return TestRewriter(semantics, config)

def rewrite_code(rewriter: TestRewriter, code: str) -> str:
    """Executes rewrite."""
    tree = cst.parse_module(code)
    try:
        new_tree = rewriter.convert(tree)
        return new_tree.code
    except Exception as e:
        pytest.fail(f"Rewrite failed: {e}")

def test_signature_injection_missing_arg(rewriter: TestRewriter) -> None:
    code = """
class Net:
    def __init__(self):
        self.layer = torch.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)
"""
    result = rewrite_code(rewriter, code)
    assert "self.layer.apply(variables, x)" in result

def test_signature_no_injection_if_present(rewriter: TestRewriter) -> None:
    code = """
class Net:
    def __init__(self):
        self.layer = torch.Linear(10, 10)

    def forward(self, variables, x):
        return self.layer(x)
"""
    result = rewrite_code(rewriter, code)
    assert "self.layer.apply(variables, x)" in result
    assert "Injected missing state argument" not in result

def test_custom_trait_injection() -> None:
    semantics = MockStateSemantics()
    config = RuntimeConfig(target_framework="mlx", strict_mode=False)
    custom_rewriter = TestRewriter(semantics, config)

    code = """
class Net:
    def __init__(self):
        self.layer = torch.Linear(10)

    def func(self, input):
        return self.layer(input)
"""
    result = rewrite_code(custom_rewriter, code)

    # Expect 'ctx' injection and 'call_fn' methods based on config
    assert "def func(self, ctx, input):" in result
    assert "self.layer.call_fn(ctx, input)" in result
EOF

cat > tests/core/rewriter/test_trait_rewriting.py << 'EOF'
"""
Tests for Trait-Based Structural Rewriting.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks import register_framework

class MockTraitSemantics(SemanticsManager):
    """
    Mock Manager that returns explicit traits.
    """

    def __init__(self):
        # Bypass super().__init__ which loads files
        self.data = {}
        self._reverse_index = {}
        self.import_data = {}
        self.test_templates = {}

        self.framework_configs = {
            "custom_nn": {
                "traits": {
                    "module_base": "custom.Layer",
                    "forward_method": "predict",
                    "requires_super_init": True,
                    "init_method_name": "__init__",
                    # Note: Both are defined, so result will have ctx AND stripped rngs
                    "inject_magic_args": [("ctx", "custom.Context")],
                    "strip_magic_args": ["rngs"],
                }
            },
            "jax": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
            # Define minimal traits for torch to prove injection works when asked
            "torch": {
                "traits": {
                    "module_base": "torch.nn.Module",
                    "requires_super_init": True,
                }
            },
            # A framework unknown to the literal code, added purely via config
            "ghost_fw": {"traits": {"module_base": "ghost.Network", "forward_method": "ghost_fwd"}},
        }

    def get_framework_config(self, framework: str) -> dict:
        return self.framework_configs.get(framework, {})

@pytest.fixture
def rewriter_factory():
    # Register dummy adapter for 'custom_nn' so RuntimeConfig validation passes during the test
    class CustomNNAdapter:
        def convert(self, x):
            return x

    register_framework("custom_nn")(CustomNNAdapter)
    register_framework("vanilla")(CustomNNAdapter)  # For fallback test
    register_framework("ghost_fw")(CustomNNAdapter)  # For dynamic detection test

    # sematics setup
    semantics = MockTraitSemantics()

    def create(target_fw):
        config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=False)
        return TestRewriter(semantics, config)

    return create

def rewrite_code(rewriter, code: str) -> str:
    tree = cst.parse_module(code)
    # Use TestRewriter wrapper
    new_tree = rewriter.convert(tree)
    return new_tree.code

def test_trait_module_inheritance_rewrite(rewriter_factory):
    rewriter = rewriter_factory("custom_nn")
    code = "class Model(torch.nn.Module): pass"
    result = rewrite_code(rewriter, code)
    assert "class Model(custom.Layer):" in result

def test_dynamic_base_discovery(rewriter_factory):
    """
    Verifies that a completely unknown framework base ('ghost.Network')
    is detected as a Module purely because it exists in the SemanticsManager config,
    without hardcoding.
    """
    # We want to convert FROM Ghost FW to Custom NN
    semantics = MockTraitSemantics()
    config = RuntimeConfig(source_framework="ghost_fw", target_framework="custom_nn", strict_mode=False)
    rewriter = TestRewriter(semantics, config)

    code = """
class MyGhost(ghost.Network):
    def forward(self, x):
        pass
"""
    result = rewrite_code(rewriter, code)

    # Base should swap to custom.Layer (from custom_nn traits)
    assert "class MyGhost(custom.Layer):" in result

    # Method should rename to 'predict' (from custom_nn traits)
    assert "def predict(self, x):" in result

def test_trait_method_renaming(rewriter_factory):
    rewriter = rewriter_factory("custom_nn")
    code = """
class Model(torch.nn.Module):
    def forward(self, x):
        pass
"""
    result = rewrite_code(rewriter, code)
    assert "def predict(self, x):" in result
    assert "def forward" not in result

def test_trait_argument_injection(rewriter_factory):
    rewriter = rewriter_factory("custom_nn")
    code = "class Model(torch.nn.Module): \n    def __init__(self): pass"
    result = rewrite_code(rewriter, code)
    assert "def __init__(self, ctx: custom.Context):" in result

def test_trait_super_init_requirement(rewriter_factory):
    rewriter = rewriter_factory("custom_nn")
    code = """
class Model(torch.nn.Module):
    def __init__(self):
        self.x = 1
"""
    result = rewrite_code(rewriter, code)
    # 'requires_super_init' is True for custom_nn
    assert "super().__init__()" in result

def test_trait_arg_stripping(rewriter_factory):
    """
    Verify behavior when stripping 'rngs' and injecting 'ctx'.
    """
    rewriter = rewriter_factory("custom_nn")
    code = """
class Model(torch.nn.Module):
    def __init__(self, rngs, x):
        pass
"""
    result = rewrite_code(rewriter, code)

    # 'rngs' should be gone. 'ctx' should be added.
    assert "def __init__(self, ctx: custom.Context, x):" in result
    assert "rngs" not in result
EOF

# 4. Cleanup tests/plugins/*.py imports
for f in tests/plugins/*.py; do
    sed -i "s/from ml_switcheroo.core.rewriter import PivotRewriter/from tests.conftest import TestRewriter as PivotRewriter/g" "$f"
done

# 5. Fix codegen test which used PivotRewriter
sed -i "s/from ml_switcheroo.core.rewriter import PivotRewriter/from tests.conftest import TestRewriter as PivotRewriter/g" tests/codegen/test_generator_shape.py

echo "✅ PivotRewriter Shim Removed and TestRewriter Created."