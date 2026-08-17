"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.graph import GraphExtractor


def get_extractor(code):
  """Docstring."""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)
  return extractor


def test_other_functions():
  """Docstring."""
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


def test_expr_not_call():
  """Docstring."""
  code = """
x = 1
1 + 1
"""
  ex = get_extractor(code)
  assert len(ex.graph.nodes) == 1


def test_unsupported_return():
  """Docstring."""
  code = """
def forward():
    return
"""
  get_extractor(code)


def test_function_names():
  """Docstring."""
  code = """
class Net:
    def setup(self):
        self.layer = op()
    def call(self, x):
        pass
"""
  get_extractor(code)


def test_other_forward_names():
  """Docstring."""
  for name in ["kernel", "f", "__call__"]:
    code = f"""
class Net:
    def {name}(self, x):
        return x
"""
    get_extractor(code)


def test_layer_def_edge_cases():
  """Docstring."""
  code = """
class Net:
    def __init__(self):
        self.layer1 = nn.Conv2d(1, bias=False)
        self.layer2 = 1
        x = nn.Linear(1)
        self.layer3 = nn.Sequential()
        self.layer3.sub = op()
"""
  get_extractor(code)


def test_data_flow_edge_cases():
  """Docstring."""
  code = """
a = 1.0
b = a
(c, d) = (1, 2)
def forward():
    x = 1
    y = z = F.relu(x)
    a.b = F.relu(x) # out_var_name is None
"""
  get_extractor(code)


def test_returns():
  """Docstring."""
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
  get_extractor(code)


def test_module_level_call():
  """Docstring."""
  code = """
F.relu(ext_var, kw=True)
"""
  get_extractor(code)


def test_resolve_layer_or_func_name_context():
  """Docstring."""
  code = """
def forward():
    F.relu(1)
"""
  ex = get_extractor(code)
  call_node = cst.parse_expression("F.relu(x)")
  # Context None
  ex._resolve_layer_or_func_name(call_node.func, context_node=None)
  # Context not call, expr, assign (hit 322 -> 325 and 325 -> 334)
  ex._resolve_layer_or_func_name(call_node.func, context_node=cst.Pass())


def test_empty_finalize():
  """Docstring."""
  ex = get_extractor("")
  assert len(ex.graph.nodes) == 0


def test_implicit_external_input():
  """Docstring."""
  code = """
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


def test_same_input_twice():
  """Docstring."""
  code = """
class Net:
    def forward(self, x):
        pass
    def call(self, x):
        pass
"""
  get_extractor(code)


def test_complex_call():
  """Docstring."""
  code = """
def forward():
    (a[0])() # hits layer_name is None in analyze_call_expression
"""
  get_extractor(code)


def test_resolve_layer_none_context():
  """Docstring."""
  from ml_switcheroo.core.graph import GraphExtractor

  ex = GraphExtractor()
  name = cst.Name("relu")
  # This will call _resolve_layer_or_func_name with context_node=None
  res = ex._resolve_layer_or_func_name(name, None)
  assert res == "func_relu"

  # also cover when context_node is some random node like cst.Pass()
  name2 = cst.Name("sigmoid")
  res2 = ex._resolve_layer_or_func_name(name2, cst.Pass())
  assert res2 == "func_sigmoid"
