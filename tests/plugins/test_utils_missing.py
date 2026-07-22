"""Test suite for the Utils Missing module."""


def test_utils_missing():
  """Verifies the behavior of utilities missing."""
  import libcst as cst
  from ml_switcheroo.plugins.utils import is_framework_module_node, _extract_root_name
  from ml_switcheroo.core.hooks import HookContext

  class DummyAlias:
    """Dummy Alias class for testing purposes."""

    def model_dump(self):
      """Mock implementation of model dump."""
      return {"name": "pd"}

  class DummyConf:
    """Dummy Conf class for testing purposes."""

    alias = DummyAlias()

  class DummySM:
    """Dummy S M class for testing purposes."""

    _source_registry = {"torch.nn": {}}
    framework_configs = {"pandas": DummyConf()}

  class DummyConfigObj:
    """Dummy Config Obj class for testing purposes."""

    source_framework = "s"
    target_framework = "t"
    effective_source = "s"
    effective_target = "t"

  ctx = HookContext(DummySM(), DummyConfigObj())
  assert is_framework_module_node(cst.Integer("1"), ctx) is False
  assert is_framework_module_node(cst.Name("pd"), ctx) is True
  assert is_framework_module_node(cst.Name("torch"), ctx) is True
  assert _extract_root_name(cst.Integer("1")) is None
