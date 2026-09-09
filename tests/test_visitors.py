"""Tests for LibCST Visitors (PyTorch, Keras, JAX, MLX)."""

import libcst as cst

from ml_switcheroo.ingestion.visitors import (
  PyTorchVisitor,
  KerasVisitor,
  JAXVisitor,
  MLXVisitor,
  ASTNormalizer,
)


def test_pytorch_visitor_state_allocations():
  """Test PyTorchVisitor extracts state allocations in __init__."""
  source_code = """
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(10))
        self.b = Parameter(torch.zeros(10))
        self.register_buffer("running_mean", torch.zeros(10))
        self.register_buffer('running_var', torch.ones(10))
        self.x = self.register_buffer('x_val', torch.zeros(10))

    def forward(self, x):
        x = x * self.w + self.b
        return x
    """
  module = cst.parse_module(source_code)
  visitor = PyTorchVisitor()
  module.visit(visitor)

  assert "w" in visitor.state_allocations
  assert "b" in visitor.state_allocations
  assert "running_mean" in visitor.state_allocations
  assert "running_var" in visitor.state_allocations
  assert "x_val" in visitor.state_allocations

  assert len(visitor.forward_nodes) == 2


def test_pytorch_visitor_non_state_allocations():
  """Test PyTorchVisitor ignores non-state allocations in __init__."""
  source_code = """
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        y = nn.Parameter(torch.zeros(10)) # Not assigned to self
        self.foo = torch.randn(10)
        self.register_buffer(123, torch.zeros(10)) # Non-string name
        self.register_buffer("no_val") # Missing value
        register_buffer("foo", torch.zeros(10)) # Call by name

        self.other_var = 5
        self.func_call = [lambda x: x][0]()
        other.register_buffer("x", torch.zeros(1))
        other.linear = nn.Parameter(torch.zeros(1))
        y = register_buffer("foo_name", torch.zeros(10)) # visit_Assign register_buffer as cst.Name
        z = register_buffer(123, torch.zeros(10)) # non-string
        w = register_buffer("no_val2") # no second arg
        self.x = register_buffer("foo_name2", torch.zeros(10)) # assigned to self

        a = 1 # Simple statement that is not Expr Call
        pass

    def other_method(self):
        pass

    def forward(self, x):
        return self.linear(x)
    """
  module = cst.parse_module(source_code)
  visitor = PyTorchVisitor()
  module.visit(visitor)

  assert len(visitor.state_allocations) == 2
  assert "foo_name" in visitor.state_allocations
  assert "foo_name2" in visitor.state_allocations
  assert len(visitor.forward_nodes) == 1


def test_keras_visitor():
  """Test KerasVisitor extraction."""
  source_code = """
class MyLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.add_weight()
        other.w = self.add_weight()

    def build(self, input_shape):
        self.b = self.add_weight()
        x = 5

    def call(self, inputs):
        return inputs * self.w + self.b

    def other(self):
        self.ignored = 5
        pass
    """
  module = cst.parse_module(source_code)
  visitor = KerasVisitor()
  module.visit(visitor)

  assert "w" in visitor.state_allocations
  assert "b" in visitor.state_allocations
  assert len(visitor.call_nodes) == 1


def test_jax_visitor():
  """Test JAXVisitor extraction."""
  source_code = """
@jax.jit
def foo(x):
    return x

@jit
def bar(x):
    return x

@jax.vmap
def baz(x):
    return x

@vmap
def qux(x):
    return x

@jax.jit(static_argnums=(0,))
def foo2(x):
    return x

@jit(static_argnums=(0,))
def bar2(x):
    return x

@jax.vmap(in_axes=(0,))
def baz2(x):
    return x

@vmap(in_axes=(0,))
def qux2(x):
    return x

@other
def norf(x):
    return x

@other(123)
def norf2(x):
    return x
    """
  module = cst.parse_module(source_code)
  visitor = JAXVisitor()
  module.visit(visitor)

  assert len(visitor.jit_nodes) == 4
  assert len(visitor.vmap_nodes) == 4


def test_mlx_visitor():
  """Test MLXVisitor extraction."""
  source_code = """
class MyModel(nn.Module):
    def __init__(self):
        self.w = mx.random.normal((10,))
        x = mx.random.normal((10,))
    """
  module = cst.parse_module(source_code)
  visitor = MLXVisitor()
  module.visit(visitor)

  assert "w" in visitor.state_allocations
  assert len(visitor.state_allocations) == 1


def test_ast_normalizer():
  """Test ASTNormalizer."""
  source_code = """
def foo():
    x = 1
    return x
    """
  module = cst.parse_module(source_code)
  transformer = ASTNormalizer()
  modified = module.visit(transformer)
  assert modified.code == source_code
