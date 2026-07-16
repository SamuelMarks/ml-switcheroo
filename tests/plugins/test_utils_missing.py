"""Auto-generated doc."""


def test_utils_missing():
  """Auto-generated doc."""
  import libcst as cst
  from ml_switcheroo.plugins.utils import is_framework_module_node, _extract_root_name
  from ml_switcheroo.core.hooks import HookContext

  class DummyAlias:
    """Auto-generated doc."""

    def model_dump(self):
      """Auto-generated doc."""
      return {"name": "pd"}

  class DummyConf:
    """Auto-generated doc."""

    alias = DummyAlias()

  class DummySM:
    """Auto-generated doc."""

    _source_registry = {"torch.nn": {}}
    framework_configs = {"pandas": DummyConf()}

  class DummyConfigObj:
    """Auto-generated doc."""

    source_framework = "s"
    target_framework = "t"
    effective_source = "s"
    effective_target = "t"

  ctx = HookContext(DummySM(), DummyConfigObj())

  # 60: empty extract root
  assert is_framework_module_node(cst.Integer("1"), ctx) is False

  # 88-90, 100-102
  assert is_framework_module_node(cst.Name("pd"), ctx) is True
  assert is_framework_module_node(cst.Name("torch"), ctx) is True

  # 113
  assert _extract_root_name(cst.Integer("1")) is None
