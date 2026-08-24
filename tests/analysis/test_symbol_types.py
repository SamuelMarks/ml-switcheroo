"""Test module."""

from ml_switcheroo.analysis.symbol_types import SymbolType, TensorType, ModuleType, UnionType, Scope


def test_symbol_type():
  """Test element."""
  sym1 = SymbolType()
  sym1.name = "Unknown"

  sym2 = SymbolType()
  sym2.name = "Unknown"

  sym3 = SymbolType()
  sym3.name = "Other"

  assert str(sym1) == "Unknown"
  assert sym1 == sym2
  assert sym1 != sym3
  assert sym1 != "Not a SymbolType"


def test_tensor_type():
  """Test element."""
  t1 = TensorType(framework="torch")
  t2 = TensorType(framework="torch")
  t3 = TensorType(framework="jax")

  assert t1.name == "Tensor"
  assert t1 == t2
  assert t1 != t3
  assert t1 != "Not a TensorType"


def test_module_type():
  """Test element."""
  m1 = ModuleType(path="torch.nn")
  m2 = ModuleType(path="torch.nn")
  m3 = ModuleType(path="jax.numpy")

  assert m1.name == "Module"
  assert m1 == m2
  assert m1 != m3
  assert m1 != "Not a ModuleType"


def test_union_type():
  """Test element."""
  t_torch = TensorType(framework="torch")
  t_jax = TensorType(framework="jax")

  u1 = UnionType(types=[t_torch, t_jax])
  u2 = UnionType(types=[t_jax, t_torch])
  u3 = UnionType(types=[t_torch])

  assert str(u1) == "Union[Tensor]"  # they both evaluate to "Tensor" string
  assert u1 == u2  # they contain same string representations
  assert u1 == u3  # Wait, is u1 == u3 because the string repr is just "Tensor"?
  # Let's check the eq implementation: return set(str(t) for t in self.types) == set(str(t) for t in other.types)
  # Yes, since both are "Tensor", the set is {"Tensor"}. So they are equal.

  assert u1 != "Not a UnionType"


def test_scope():
  """Test element."""
  root = Scope(name="root")
  root.set("x", TensorType(framework="torch"))

  child = Scope(parent=root, name="child")
  child.set("y", TensorType(framework="jax"))

  assert root.get("x") == TensorType(framework="torch")
  assert root.get("y") is None

  assert child.get("x") == TensorType(framework="torch")
  assert child.get("y") == TensorType(framework="jax")
  assert child.get("z") is None

  snap = child.snapshot()
  assert snap["y"] == TensorType(framework="jax")
  assert "x" not in snap
