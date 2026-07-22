"""Test suite for the Infix Transform module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInfixSemantics(SemanticsManager):
  """Mock Infix Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockInfixSemantics instance."""
    super().__init__()
    self.data = {}
    self._reverse_index = {}

    def inject(name, s_api, op, arity=2):
      """Mock implementation of inject."""
      args = ["x", "y"] if arity == 2 else ["x"]
      variants = {"torch": {"api": s_api}, "jax": {"transformation_type": "infix", "operator": op}}
      self.data[name] = {"variants": variants, "std_args": args}
      self._reverse_index[s_api] = (name, self.data[name])

    inject("div", "torch.div", "/")
    inject("add", "torch.add", "+")
    inject("sub", "torch.sub", "-")
    inject("mul", "torch.mul", "*")
    inject("pow", "torch.pow", "**")
    inject("matmul", "torch.matmul", "@")
    inject("bit_and", "torch.bitwise_and", "&")
    inject("bit_or", "torch.bitwise_or", "|")
    inject("bit_xor", "torch.bitwise_xor", "^")
    inject("lshift", "torch.left_shift", "<<")
    inject("rshift", "torch.right_shift", ">>")
    inject("mod", "torch.fmod", "%")
    inject("neg", "torch.neg", "-", arity=1)
    inject("invert", "torch.bitwise_not", "~", arity=1)
    inject("logical_not", "torch.logical_not", "not", arity=1)
    inject("bad_op", "torch.bad", "???")


@pytest.fixture
def engine():
  """Provides a mock engine for testing."""
  return ASTEngine(semantics=MockInfixSemantics(), source="torch", target="jax", strict_mode=True)


def test_infix_arithmetic_ops(engine):
  """Verifies the behavior of infix arithmetic ops."""
  assert "a / b" in engine.run("torch.div(a, b)").code
  assert "a + b" in engine.run("torch.add(a, b)").code
  assert "a - b" in engine.run("torch.sub(a, b)").code
  assert "a * b" in engine.run("torch.mul(a, b)").code
  assert "a ** b" in engine.run("torch.pow(a, b)").code
  assert "a % b" in engine.run("torch.fmod(a, b)").code


def test_infix_bitwise_ops(engine):
  """Verifies the behavior of infix bitwise ops."""
  assert "a & b" in engine.run("torch.bitwise_and(a, b)").code
  assert "a | b" in engine.run("torch.bitwise_or(a, b)").code
  assert "a ^ b" in engine.run("torch.bitwise_xor(a, b)").code
  assert "a << b" in engine.run("torch.left_shift(a, b)").code
  assert "a >> b" in engine.run("torch.right_shift(a, b)").code


def test_infix_matmul_reordering(engine):
  """Verifies the behavior of infix matmul reordering."""
  code = "res = torch.matmul(y=mat_b, x=mat_a)"
  result = engine.run(code)
  assert "mat_a @ mat_b" in result.code


def test_unary_operators(engine):
  """Verifies the behavior of unary operators."""
  assert "-x" in engine.run("torch.neg(x)").code
  assert "~x" in engine.run("torch.bitwise_not(x)").code
  assert "not x" in engine.run("torch.logical_not(x)").code


def test_unary_complex_expression_parens(engine):
  """Verifies the behavior of unary complex expression parens."""
  code = "y = torch.neg(a + b)"
  result = engine.run(code)
  assert "-(a + b)" in result.code


def test_infix_invalid_arg_count_binary(engine):
  """Verifies the behavior of infix invalid argument count binary."""
  code = "res = torch.div(start_val)"
  result = engine.run(code)
  assert "torch.div(start_val)" in result.code
  assert "args=1" in str(result.errors) or len(result.errors) > 0
  assert "# Reason: Infix/Prefix transformation failed" in result.code


def test_infix_invalid_operator_symbol(engine):
  """Verifies the behavior of infix invalid operator symbol."""
  code = "res = torch.bad(a, b)"
  result = engine.run(code)
  assert "torch.bad(a, b)" in result.code
  assert "# Reason: Infix/Prefix transformation failed" in result.code
  assert "Unsupported binary operator: ???" in result.code
