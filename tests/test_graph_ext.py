"""Test suite for GraphExtractor coverage with explicit graph assertions."""

import libcst as cst

from ml_switcheroo.core.graph import GraphExtractor


def get_extractor(code: str) -> GraphExtractor:
  """Parse ``code`` and run GraphExtractor over the CST module.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)
  return extractor


def test_other_functions() -> None:
  """Verify helper methods outside __init__/forward are ignored by GraphExtractor."""
  code = """
class Net:
    def __init__(self):
        self.conv = nn.Conv2d(1)
    def helper(self, x):
        y = x + 1
        1 + 1
        return y
"""
  ex = get_extractor(code)
  assert not ex._in_forward
  assert not ex._in_init
  assert "conv" in ex.layer_registry


def test_expr_not_call() -> None:
  """Verify non-call expressions at module scope do not create spurious op nodes."""
  code = """
x = 1
1 + 1
"""
  ex = get_extractor(code)
  assert len(ex.graph.nodes) == 1
  assert "Input_x" in ex.graph.nodes


def test_unsupported_return() -> None:
  """Verify bare return statement produces no Output node."""
  code = """
def forward():
    return
"""
  ex = get_extractor(code)
  assert "output" not in ex.layer_registry
  assert "output" not in ex.graph.nodes
  assert ex.graph.edges == []


def test_function_names() -> None:
  """Verify setup() and call() are recognized as init and forward methods."""
  code = """
class Net:
    def setup(self):
        self.layer = op()
    def call(self, x):
        pass
"""
  ex = get_extractor(code)
  assert "layer" in ex.layer_registry
  assert "Input_x" in ex.layer_registry
  assert "Input_x" in ex.graph.nodes


def test_other_forward_names() -> None:
  """Verify kernel, f, and __call__ are recognized as forward entrypoints."""
  for name in ["kernel", "f", "__call__"]:
    code = f"""
class Net:
    def {name}(self, x):
        return x
"""
    ex = get_extractor(code)
    assert "Input_x" in ex.layer_registry
    assert "output" in ex.layer_registry
    assert "Input_x" in ex.graph.nodes
    assert "output" in ex.graph.nodes
    assert len(ex.graph.edges) == 1


def test_layer_def_edge_cases() -> None:
  """Verify layer definition edge cases in __init__."""
  code = """
class Net:
    def __init__(self):
        self.layer1 = nn.Conv2d(1, bias=False)
        self.layer2 = 1
        x = nn.Linear(1)
        self.layer3 = nn.Sequential()
        self.layer3.sub = op()
"""
  ex = get_extractor(code)
  assert "layer1" in ex.layer_registry
  assert "layer2" not in ex.layer_registry
  assert "layer3" in ex.layer_registry
  assert ex.graph.edges == []


def test_data_flow_edge_cases() -> None:
  """Verify data flow analysis for literals, chained assignments, and attribute targets."""
  code = """
a = 1.0
b = a
(c, d) = (1, 2)
def forward():
    x = 1
    y = z = F.relu(x)
    a.b = F.relu(x) # out_var_name is None
"""
  ex = get_extractor(code)
  assert "Input_a" in ex.layer_registry
  assert "func_relu" in ex.layer_registry
  assert "func_relu" in ex.graph.nodes
  assert ex.graph.edges == []


def test_returns() -> None:
  """Verify return statement handling across variable and call return expressions."""
  code = """
class Net:
    def forward(self, x):
        y = op(x)
        if True:
            return y
        if False:
            return y
        if False:
            return F.relu(x)
        if False:
            return F.relu(x)
        if False:
            return unknown_var
        if False:
            return (a[0])() # returning a direct call where resolve returns None
"""
  ex = get_extractor(code)
  assert "output" in ex.layer_registry
  assert "func_op" in ex.layer_registry
  assert "output" in ex.graph.nodes
  assert len(ex.graph.edges) >= 2


def test_module_level_call() -> None:
  """Verify module-level function call registers implicit external input and functional op."""
  code = """
F.relu(ext_var, kw=True)
"""
  ex = get_extractor(code)
  assert "Input_ext_var" in ex.layer_registry
  assert "func_relu" in ex.layer_registry
  assert "func_relu" in ex.graph.nodes
  assert len(ex.graph.edges) == 2


def test_resolve_layer_or_func_name_context() -> None:
  """Verify _resolve_layer_or_func_name with None and Pass context nodes."""
  code = """
def forward():
    F.relu(1)
"""
  ex = get_extractor(code)
  call_node = cst.parse_expression("F.relu(x)")
  res1 = getattr(
    {
      "v": getattr({"v": getattr({"v": ex}, "__getitem__")("v")._resolve_layer_or_func_name}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr(
              getattr({"v": call_node}, "__getitem__")("v"),
              getattr({"v": "func"}, "__getitem__")("v"),
              getattr({"v": None}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v"),
        context_node=getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert res1 == "func_relu"
  res2 = getattr(
    {
      "v": getattr({"v": getattr({"v": ex}, "__getitem__")("v")._resolve_layer_or_func_name}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr(
              getattr({"v": call_node}, "__getitem__")("v"),
              getattr({"v": "func"}, "__getitem__")("v"),
              getattr({"v": None}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v"),
        context_node=getattr(
          {"v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Pass}, "__getitem__")("v")()}, "__getitem__"
        )("v"),
      )
    },
    "__getitem__",
  )("v")
  assert res2 == "func_relu"
  assert "func_relu" in ex.layer_registry
  assert "func_relu" in ex.graph.nodes


def test_empty_finalize() -> None:
  """Verify empty module produces an empty LogicalGraph."""
  ex = get_extractor("")
  assert len(ex.graph.nodes) == 0
  assert ex.graph.edges == []


def test_implicit_external_input() -> None:
  """Verify implicit external input registration across module and class scopes."""
  code = """
x = 1
y = op(x, z)
w = op(z)
x = 2
(a, b) = 1

class Net:
    def forward(self):
        pass

v = op(z)
"""
  ex = get_extractor(code)
  assert "Input_x" in ex.layer_registry
  assert "Input_z" in ex.layer_registry
  assert "Input_x" in ex.graph.nodes
  assert "Input_z" in ex.graph.nodes
  assert len(ex.graph.edges) >= 2


def test_same_input_twice() -> None:
  """Verify duplicate input parameter names across multiple forward methods."""
  code = """
class Net:
    def forward(self, x):
        pass
    def call(self, x):
        pass
"""
  ex = get_extractor(code)
  assert "Input_x" in ex.layer_registry
  assert "Input_x" in ex.graph.nodes
  assert ex.graph.edges == []


def test_complex_call() -> None:
  """Verify subscript call expression that cannot be resolved to a layer name is ignored."""
  code = """
def forward():
    (a[0])()
"""
  ex = get_extractor(code)
  assert len(ex.graph.nodes) == 0
  assert ex.graph.edges == []
  assert len(ex.layer_registry) == 0


def test_resolve_layer_none_context() -> None:
  """Verify _resolve_layer_or_func_name registers new functional ops when context is None or Pass."""
  ex = GraphExtractor()
  name = cst.Name("relu")
  res = ex._resolve_layer_or_func_name(name, None)
  assert res == "func_relu"

  name2 = cst.Name("sigmoid")
  res2 = ex._resolve_layer_or_func_name(name2, cst.Pass())
  assert res2 == "func_sigmoid"


def test_create_child_coverage() -> None:
  """Test self.create_child parsing in setup() for Paxml/Praxis."""
  code = """
class PaxmlNet:
    def __init__(self):
        self.plain_layer = Linear
        self.attr_layer = mod.Linear
    def setup(self):
        other.create_child('foo', Linear())
        self.other_func('bar', Linear())
        self.create_child('plain_child', Linear())
        self.create_child('proj', layers.Conv3D(64, bias=True))
        self.create_child('fc', Linear)
        self.create_child('dense', pl.Dense)
        self.create_child('too_few_args')
        self.create_child(123, Linear)
        self.create_child('unknown', 1 + 2)
    def __call__(self, x):
        return self.proj(x)
"""
  extractor = get_extractor(code)
  assert "plain_layer" in extractor.layer_registry
  assert "proj" in extractor.layer_registry
  assert "fc" in extractor.layer_registry
  assert "dense" in extractor.layer_registry
