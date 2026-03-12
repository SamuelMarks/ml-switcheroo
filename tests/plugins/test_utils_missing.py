def test_utils_missing():
  import libcst as cst
  from ml_switcheroo.plugins.utils import is_framework_module_node, _extract_root_name
  from ml_switcheroo.core.hooks import HookContext

  class DummyAlias:
    def model_dump(self):
      return {"name": "pd"}

  class DummyConf:
    alias = DummyAlias()

  class DummySM:
    _source_registry = {"torch.nn": {}}
    framework_configs = {"pandas": DummyConf()}

  class DummyConfigObj:
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
