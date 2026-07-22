"""Test suite for the Api Call Mixin module."""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter.passes.api_call_mixin import ApiTransformerCallMixin


class MockTracer:
  """Mock Tracer class for testing purposes."""

  def log_inspection(self, *args, **kwargs):
    """Mock implementation of log inspection."""
    pass

  def log_mutation(self, *args, **kwargs):
    """Mock implementation of log mutation."""
    pass


class MockHookContext:
  """Mock Hook Context class for testing purposes."""

  def __init__(self):
    """Initializes the MockHookContext instance."""
    self.current_op_id = None


class MockContext:
  """Mock Context class for testing purposes."""

  def __init__(self):
    """Initializes the MockContext instance."""
    self.hook_context = MockHookContext()
    self.symbol_table = None


class MockTransformer(ApiTransformerCallMixin, cst.CSTTransformer):
  """Mock Transformer class for testing purposes."""

  def __init__(self):
    """Initializes the MockTransformer instance."""
    self.strict_mode = True
    self.source_fw = "torch"
    self.target_fw = "jax"
    self.warnings = []
    self.failures = []
    self.semantics = self
    self.context = MockContext()

  def get_definition(self, name):
    """Mock implementation of get definition."""
    if name == "torch.known":
      return ("known_id", {"deprecated": True, "replaced_by": "other"})
    if name == "torch.nodep":
      return ("nodep_id", {})
    return None

  def _get_qualified_name(self, node):
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name):
      return f"{node.value.value}.{node.attr.value}"
    return None

  def _get_mapping(self, name, silent=False):
    if name == "torch.known":
      return {"api": "jnp.known"}
    if name == "torch.nodep":
      return {"api": "jnp.nodep", "min_version": "1.0", "max_version": "2.0"}
    return None

  def check_version_constraints(self, min_v, max_v):
    """Mock implementation of check version constraints."""
    if min_v == "1.0":
      return "Version warning"
    return None

  def _report_warning(self, msg):
    self.warnings.append(msg)

  def _report_failure(self, msg):
    self.failures.append(msg)

  def _is_module_alias(self, node):
    return False

  def _handle_variant_imports(self, mapping):
    pass

  def _create_name_node(self, name):
    return cst.Name(name)


def test_leave_Call_pre_check():
  """Verifies the behavior of leave Call pre check."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "handle_pre_checks", lambda *args: (True, cst.Call(func=cst.Name("handled"), args=[])))
    tree = cst.parse_module("foo()")
    new_tree = tree.visit(transformer)
    assert getattr(new_tree.body[0].body[0].value.func, "value", None) == "handled"


def test_leave_Call_no_mapping_super():
  """Verifies the behavior of leave Call no mapping super."""
  transformer = MockTransformer()
  tree = cst.parse_module("super()")
  new_tree = tree.visit(transformer)
  assert new_tree.body[0].body[0].value.func.value == "super"


def test_leave_Call_no_mapping_strict():
  """Verifies the behavior of leave Call no mapping strict."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "get_tracer", lambda: MockTracer())
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    tree = cst.parse_module("torch.unknown()")
    tree.visit(transformer)
    assert len(transformer.failures) == 1
    assert "not found in semantics" in transformer.failures[0]


def test_leave_Call_with_mapping_deprecated():
  """Verifies the behavior of leave Call with mapping deprecated."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("strategy_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("torch.known()")
    new_tree = tree.visit(transformer)
    assert len(transformer.warnings) == 1
    assert "Usage of deprecated" in transformer.warnings[0]
    assert new_tree.body[0].body[0].value.func.value == "strategy_ok"


def test_leave_Call_with_mapping_version_warn():
  """Verifies the behavior of leave Call with mapping version warn."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("strategy_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("torch.nodep()")
    new_tree = tree.visit(transformer)
    assert len(transformer.warnings) == 1
    assert transformer.warnings[0] == "Version warning"
    assert new_tree.body[0].body[0].value.func.value == "strategy_ok"


def test_leave_Call_implicit_method():
  """Verifies the behavior of leave Call implicit method."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "resolve_implicit_method", lambda *args: "torch.nodep")
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("implicit_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("x.nodep()")
    new_tree = tree.visit(transformer)
    assert new_tree.body[0].body[0].value.func.value == "implicit_ok"
