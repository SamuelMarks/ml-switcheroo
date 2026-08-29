"""Module docstring."""

import libcst as cst

from ml_switcheroo.core.graph import GraphExtractor


def get_extractor(code: str) -> GraphExtractor:
  """Docstring."""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)
  return extractor


def test_other_functions() -> None:
  """Docstring."""
  code: str = """
class Net:
    def __init__(self):
        self.conv = nn.Conv2d(1)
    def helper(self, x):
        y = x + 1
        1 + 1
        return y
"""
  ex: GraphExtractor = get_extractor(code)
  assert not ex._in_forward
  assert not ex._in_init


def test_expr_not_call() -> None:
  """Docstring."""
  code: str = """
x = 1
1 + 1
"""
  ex: GraphExtractor = get_extractor(code)
  assert len(ex.graph.nodes) == 1


def test_unsupported_return() -> None:
  """Docstring."""
  code: str = """
def forward():
    return
"""
  get_extractor(code)


def test_function_names() -> None:
  """Docstring."""
  code: str = """
class Net:
    def setup(self):
        self.layer = op()
    def call(self, x):
        pass
"""
  get_extractor(code)


def test_other_forward_names() -> None:
  """Docstring."""
  for name in ["kernel", "f", "__call__"]:
    code: str = f"""
class Net:
    def {name}(self, x):
        return x
"""
    get_extractor(code)


def test_layer_def_edge_cases() -> None:
  """Docstring."""
  code: str = """
class Net:
    def __init__(self):
        self.layer1 = nn.Conv2d(1, bias=False)
        self.layer2 = 1
        x = nn.Linear(1)
        self.layer3 = nn.Sequential()
        self.layer3.sub = op()
"""
  get_extractor(code)


def test_data_flow_edge_cases() -> None:
  """Docstring."""
  code: str = """
a = 1.0
b = a
(c, d) = (1, 2)
def forward():
    x = 1
    y = z = F.relu(x)
    a.b = F.relu(x) # out_var_name is None
"""
  get_extractor(code)


def test_returns() -> None:
  """Docstring."""
  code: str = """
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
  get_extractor(code)


def test_module_level_call() -> None:
  """Docstring."""
  code: str = """
F.relu(ext_var, kw=True)
"""
  get_extractor(code)


def test_resolve_layer_or_func_name_context() -> None:
  """Docstring."""
  code: str = """
def forward():
    F.relu(1)
"""
  ex: GraphExtractor = get_extractor(code)
  call_node: cst.BaseExpression = cst.parse_expression("F.relu(x)")
  # Context None
  ex._resolve_layer_or_func_name(getattr(call_node, "func", None), context_node=None)
  # Context not call, expr, assign (hit 322 -> 325 and 325 -> 334)
  ex._resolve_layer_or_func_name(getattr(call_node, "func", None), context_node=cst.Pass())


def test_empty_finalize() -> None:
  """Docstring."""
  ex: GraphExtractor = get_extractor("")
  assert len(ex.graph.nodes) == 0


def test_implicit_external_input() -> None:
  """Docstring."""
  code: str = """
x = 1
y = op(x, z)
w = op(z)
x = 2
(a, b) = 1

class Net:
    def forward(self):
        pass

# Net.forward resets provenance to {}.
# Now Input_z is in layer_registry but z is not in provenance.
# This hits the false branch for ext_id not in self.layer_registry.
v = op(z)
"""
  get_extractor(code)


def test_same_input_twice() -> None:
  """Docstring."""
  code: str = """
class Net:
    def forward(self, x):
        pass
    def call(self, x):
        pass
"""
  get_extractor(code)


def test_complex_call() -> None:
  """Docstring."""
  code: str = """
def forward():
    (a[0])() # hits layer_name is None in analyze_call_expression
"""
  get_extractor(code)


def test_resolve_layer_none_context() -> None:
  """Docstring."""
  from ml_switcheroo.core.graph import GraphExtractor

  ex: GraphExtractor = GraphExtractor()
  name: cst.Name = cst.Name("relu")
  # This will call _resolve_layer_or_func_name with context_node=None
  res: str = ex._resolve_layer_or_func_name(name, None)
  assert res == "func_relu"

  # also cover when context_node is some random node like cst.Pass()
  name2: cst.Name = cst.Name("sigmoid")
  res2: str = ex._resolve_layer_or_func_name(name2, cst.Pass())
  assert res2 == "func_sigmoid"
