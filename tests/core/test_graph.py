"""Unit tests for `GraphExtractor` (`ml_switcheroo.core.graph`)."""

import libcst as cst

from ml_switcheroo.core.graph import GraphExtractor

_NL = chr(10)


def extract_from_code(code: str) -> GraphExtractor:
  """Parse Python `code` and visit it with `GraphExtractor`.

  Args:
      code: Python source code snippet.

  Returns:
      Populated `GraphExtractor` instance.
  """
  extractor = GraphExtractor()
  cst.parse_module(code).visit(extractor)
  return extractor


def test_init_and_setup_layer_registration() -> None:
  """Verify `__init__` and `setup` register `self.<layer>` and `self.create_child(...)` nodes."""
  code = """
class PaxmlNet:
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, bias=False)
        self.conv_plain = Conv2d(8, 16)
        self.plain_layer = Linear
        self.attr_layer = mod.Linear
        self.scalar_val = 1
        local_unused = nn.Linear(1)
        self.conv1.sub = op()
        other.layer = nn.Linear()

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

    def helper(self, x):
        y = x + 1
        1 + 1
        return y
"""
  ex = extract_from_code(code)
  assert ex.model_name == "PaxmlNet"
  assert {"conv1", "plain_layer", "attr_layer", "plain_child", "proj", "fc", "dense"} <= set(ex.layer_registry.keys())
  assert "scalar_val" not in ex.layer_registry
  conv = ex.layer_registry["conv1"]
  assert conv.op_type == "Conv2d"
  assert conv.attributes == {"arg_0": "16", "arg_1": "32", "kernel_size": "3", "bias": "False"}


def test_forward_entrypoints_and_multi_call_dataflow() -> None:
  """Verify `forward`, `call`, `__call__`, `kernel`, and `f` trace inputs, repeated ops, and returns."""
  code = """
class MultiReluNet:
    def __init__(self):
        self.conv = nn.Conv2d(1, 20, 5)

    def forward(self, x):
        # First conv comment
        h0 = self.conv(x, x.size(0))  # inline conv1
        # Second conv comment
        h0b = self.conv(h0)  # inline conv2
        h1 = F.relu(h0b)
        h2 = F.relu(h1)
        h_cat = F.cat([h2, x])
        h_res = h_cat + x
        return F.relu(h_res)
"""
  tree = cst.parse_module(code)
  ex = GraphExtractor()
  tree.visit(ex)

  assert {"Input_x", "conv", "func_relu", "func_relu_1", "func_relu_2", "output"} <= set(ex.layer_registry.keys())
  edges = [(e.source, e.target) for e in ex.graph.edges]
  assert ("Input_x", "conv") in edges
  assert ("conv", "func_relu") in edges
  assert ("func_relu_2", "output") in edges

  # Re-resolving an already-tracked call AST node returns its existing `func_relu` ID
  cls_node = cst.ensure_type(tree.body[0], cst.ClassDef)
  fwd_fn = cst.ensure_type(cls_node.body.body[1], cst.FunctionDef)
  relu_stmt = cst.ensure_type(fwd_fn.body.body[2], cst.SimpleStatementLine)
  relu_assign = cst.ensure_type(relu_stmt.body[0], cst.Assign)
  relu_call = cst.ensure_type(relu_assign.value, cst.Call)
  assert ex._resolve_layer_or_func_name(relu_call.func, context_node=relu_assign) == "func_relu"

  for entry_name in ("call", "__call__", "kernel", "f"):
    entry_ex = extract_from_code(_NL.join(["class Net:", f"    def {entry_name}(self, x):", "        return x"]))
    assert "Input_x" in entry_ex.graph.nodes
    assert "output" in entry_ex.graph.nodes
    assert len(entry_ex.graph.edges) == 1

  dup_in = extract_from_code(
    _NL.join(["class Net:", "    def forward(self, x):", "        pass", "    def call(self, x):", "        pass"])
  )
  assert "Input_x" in dup_in.graph.nodes


def test_top_level_flow_returns_and_unresolvable_edge_cases() -> None:
  """Verify module-level dataflow, bare/complex returns, chained targets, and unresolvable calls."""
  top_ex = extract_from_code(
    """
x = 1
arr[0] = 2
b = x
(u, v) = (1, 2)
1 + 1
y = add(x, ext_var, kw=True)
F.relu(ext_var)
x = 2
"""
  )
  assert {"Input_x", "Input_ext_var", "func_add", "func_relu"} <= set(top_ex.graph.nodes.keys())

  from ml_switcheroo.core.compiler.ir import LogicalNode

  pre_ex = GraphExtractor()
  pre_ex.layer_registry["Input_pre"] = LogicalNode("Input_pre", "Input", attributes={"name": "pre"})
  cst.parse_module("out = fn(pre)").visit(pre_ex)
  assert pre_ex.provenance["pre"] == "Input_pre"

  ret_ex = extract_from_code(
    """
class Net:
    def __init__(self):
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        u = self.unregistered_layer(x)
        y = z = F.relu(x)
        a.b = F.relu(x)
        x[0] = func(y[0])
        if True:
            return
        if False:
            return x + 1
        if False:
            return y
        if False:
            return y
        if False:
            return self.fc(x)
        if False:
            return F.relu(x)
        if False:
            return unknown_var
        return (a[0])()
"""
  )
  assert "output" in ret_ex.graph.nodes

  empty_ex = extract_from_code("")
  assert empty_ex.graph.nodes == {}
  assert empty_ex.graph.edges == []

  # Direct helper edge cases (`context_node=None`, `cst.Pass()`, `cst.Call`, unresolvable call)
  helper_ex = GraphExtractor()
  helper_ex._in_forward = True
  helper_ex._scope_depth = 1
  cst.parse_module(f"x = y{_NL}x = func(kwarg=1)").visit(helper_ex)

  call_node = cst.ensure_type(cst.parse_expression("some_func(1)"), cst.Call)
  assert helper_ex._resolve_layer_or_func_name(call_node.func, context_node=call_node) == "func_some_func"
  assert helper_ex._resolve_layer_or_func_name(cst.Name("relu"), context_node=None) == "func_relu"
  assert helper_ex._resolve_layer_or_func_name(cst.Name("sigmoid"), context_node=cst.Pass()) == "func_sigmoid"
  self_attr = cst.ensure_type(cst.parse_expression("self.conv"), cst.Attribute)
  assert helper_ex._resolve_layer_or_func_name(self_attr, context_node=None) == "conv"

  bad_call = cst.ensure_type(cst.parse_expression("func()[0]()"), cst.Call)
  helper_ex._analyze_call_expression(bad_call, [])
