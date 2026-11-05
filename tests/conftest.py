"""Test suite for the Conftest module."""

import importlib
import inspect
import sys
import types
import typing
import warnings
from pathlib import Path
from unittest import mock

import libcst as cst
import pydantic
import pytest

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter import ApiPass, AuxiliaryPass, RewriterContext, RewriterPipeline, StructuralPass
from ml_switcheroo.core.rewriter.context import SemanticsProtocol
from ml_switcheroo.semantics.manager import SemanticsManager

warnings.filterwarnings("ignore", message=".*np\\.object.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow\\.lite.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax\\._src\\.deprecations.*")
src_path: Path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
try:
  importlib.import_module("ml_switcheroo.frameworks")
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY
except ImportError:
  _ADAPTER_REGISTRY = {}


class TestRewriter:
  """Test helper class TestRewriter."""

  __test__ = False

  def __init__(
    self,
    semantics: SemanticsManager | SemanticsProtocol,
    config: RuntimeConfig | None = None,
    symbol_table: SymbolTable | None = None,
  ) -> None:
    """Initialize test helper instance.

    Args:
        semantics: SemanticsManager instance providing operation and trait definitions.
        config: RuntimeConfig specifying source and target frameworks.
        symbol_table: SymbolTable tracking variable types and aliases.
    """
    self.context = RewriterContext(semantics, config, symbol_table)
    self.pipeline = RewriterPipeline([StructuralPass(), ApiPass(), AuxiliaryPass()])

  @property
  def ctx(self) -> HookContext:
    """Helper to ctx.

    Returns:
        The `ctx` property value (HookContext).
    """
    return self.context.hook_context

  @property
  def semantics(self) -> SemanticsManager:
    """Helper to semantics.

    Returns:
        The `semantics` property value (SemanticsManager).
    """
    return self.context.semantics

  def convert(self, module: cst.Module) -> cst.Module:
    """Converts .

    Args:
        module: The module (cst.Module) for convert.

    Returns:
        Transpiled source code string or rewritten CST module.
    """
    return self.pipeline.run(module, self.context)


class SnapshotAssert:
  """Test helper class SnapshotAssert."""

  def __init__(self, request: pytest.FixtureRequest) -> None:
    """Initializes the SnapshotAssert instance.

    Args:
        request: Pytest fixture request context.
    """
    self.request = request
    self.test_name: str = request.node.name
    self.module_path: Path = Path(request.node.fspath).parent
    self.snapshot_dir: Path = self.module_path / "__snapshots__"
    self.update_mode: bool = request.config.getoption("--update-snapshots", default=False)

  def assert_match(
    self, content: str, extension: str = "txt", normalizer: typing.Callable[[str], str] | None = None
  ) -> None:
    """Helper to assert match.

    Args:
        content: The content (str) for assert match.
        extension: The extension (str) for assert match.
        normalizer: The normalizer (typing.Callable[[str], str] | None) for assert match.
    """
    if not self.snapshot_dir.exists():
      self.snapshot_dir.mkdir(parents=True)
    snapshot_file: Path = self.snapshot_dir / f"{self.test_name}.{extension}"
    content = content.replace("\r\n", "\n")
    if self.update_mode or not snapshot_file.exists():
      normalized_to_write: str = normalizer(content) if normalizer else content
      snapshot_file.write_text(normalized_to_write, encoding="utf-8")
      if self.update_mode:
        return
    expected: str = snapshot_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    lhs: str = content
    rhs: str = expected
    if normalizer:
      lhs = normalizer(lhs)
      rhs = normalizer(rhs)
    assert lhs == rhs, f"Snapshot mismatch for {snapshot_file.name}. Run with --update-snapshots to accept changes."


@pytest.fixture
def snapshot(request: pytest.FixtureRequest) -> SnapshotAssert:
  """Execute snapshot helper logic.

  Args:
      request: Pytest fixture request context.

  Returns:
      Configured `snapshot` fixture value (SnapshotAssert).
  """
  return SnapshotAssert(request)


@pytest.fixture(autouse=True)
def isolate_semantics_manager() -> typing.Generator[None, None, None]:
  """Helper to isolate semantics manager.

  Returns:
      Configured `isolate_semantics_manager` fixture value (typing.Generator[None, None, None]).
  """
  from ml_switcheroo.semantics import manager as manager_module

  original_cache = manager_module._MANAGER_CACHE
  manager_module._MANAGER_CACHE = None
  yield
  manager_module._MANAGER_CACHE = original_cache


@pytest.fixture(autouse=True)
def isolate_hook_registry() -> typing.Generator[None, None, None]:
  """Helper to isolate hook registry.

  Returns:
      Configured `isolate_hook_registry` fixture value (typing.Generator[None, None, None]).
  """
  import ml_switcheroo.core.hooks as hooks_module
  import ml_switcheroo.core.hooks_registry as hr_module
  from ml_switcheroo.core.hooks import _HOOK_METADATA, _HOOKS

  original_hooks = _HOOKS.copy()
  original_metadata = _HOOK_METADATA.copy()
  original_loaded = hr_module._PLUGINS_LOADED
  yield
  _HOOKS.clear()
  _HOOKS.update(original_hooks)
  _HOOK_METADATA.clear()
  _HOOK_METADATA.update(original_metadata)
  hr_module._PLUGINS_LOADED = original_loaded
  hooks_module._PLUGINS_LOADED = original_loaded


@pytest.fixture(autouse=True)
def isolate_framework_registry() -> typing.Generator[None, None, None]:
  """Helper to isolate framework registry.

  Returns:
      Configured `isolate_framework_registry` fixture value (typing.Generator[None, None, None]).
  """
  original_registry = _ADAPTER_REGISTRY.copy()
  yield
  _ADAPTER_REGISTRY.clear()
  _ADAPTER_REGISTRY.update(original_registry)


def pytest_addoption(parser: pytest.Parser) -> None:
  """Execute pytest addoption helper logic.

  Args:
      parser: Parser instance under test.
  """
  parser.addoption("--update-snapshots", action="store_true", default=False, help="Update snapshots for visual tests")


class _MissingSentinel:
  """Sentinel class for missing default arguments in test helpers."""


_MISSING_SENTINEL = _MissingSentinel()

_ObjT = typing.TypeVar("_ObjT")
_ValT = typing.TypeVar("_ValT")


class _Box:
  """Typed box holding an attribute value for test assertions."""

  def __init__(self, v: _ValT) -> None:
    """Initialize box with value.

    Args:
        v: Stored value.
    """
    self.v = v


def _unbox(val: _ValT) -> mock.MagicMock:
  """Return value typed as MagicMock for test helper assertions.

  Args:
      val: Value to unbox.

  Returns:
      Unboxed value.
  """
  res: mock.MagicMock = inspect.getattr_static(_Box(val), "v")
  return res


def _test_getattr(
  obj: _ObjT,
  name: str,
  default: _ValT | _MissingSentinel = _MISSING_SENTINEL,
) -> mock.MagicMock:
  """Retrieve attribute ``name`` from ``obj`` using ``inspect.getattr_static`` for test assertions.

  Args:
      obj: Target object to inspect.
      name: Attribute name to retrieve.
      default: Optional fallback value if attribute is not found.

  Returns:
      Resolved attribute value or ``default``.
  """
  if isinstance(obj, dict) and name == "__getitem__":
    return _unbox(lambda k: obj[k])
  if isinstance(obj, dict) and name == "__setitem__":

    def _set_item(k: _ObjT, v: _ValT) -> None:
      """Assign key ``k`` to value ``v`` in ``obj``.

      Args:
          k: Dictionary key.
          v: Value to store.
      """
      obj[k] = v

    return _unbox(_set_item)
  if callable(obj) and name == "__call__":
    return _unbox(obj)
  try:
    raw: mock.MagicMock = inspect.getattr_static(obj, name)
    if isinstance(raw, property) and raw.fget is not None:
      res_prop: mock.MagicMock = raw.fget(obj)
      return res_prop
    if isinstance(raw, classmethod):
      cls_target = obj if isinstance(obj, type) else type(obj)
      return _unbox(types.MethodType(raw.__func__, cls_target))
    if isinstance(raw, staticmethod):
      return _unbox(raw.__func__)
    if (inspect.ismemberdescriptor(raw) or inspect.isgetsetdescriptor(raw)) and not isinstance(
      obj, (type, types.ModuleType)
    ):
      get_fn = inspect.getattr_static(raw, "__get__")
      return _unbox(get_fn(raw, obj, type(obj)))
    if inspect.ismethoddescriptor(raw) and not isinstance(obj, (type, types.ModuleType)):
      return _unbox(types.MethodType(raw, obj))
    if inspect.isfunction(raw) and not isinstance(obj, (type, types.ModuleType)) and not inspect.isroutine(obj):
      return _unbox(types.MethodType(raw, obj))
    return raw
  except AttributeError:
    if isinstance(obj, mock.NonCallableMock):
      mock_children = inspect.getattr_static(obj, "_mock_children")
      if isinstance(mock_children, dict) and name in mock_children:
        val_mc = mock_children[name]
        if type(val_mc).__name__ == "_SentinelObject":
          if not isinstance(default, _MissingSentinel):
            return _unbox(default)
          raise
        return _unbox(val_mc)
      mock_methods = inspect.getattr_static(obj, "_mock_methods")
      if isinstance(mock_methods, (list, tuple)) and name not in mock_methods:
        if not isinstance(default, _MissingSentinel):
          return _unbox(default)
        raise
      get_child = inspect.getattr_static(obj, "_get_child_mock")
      child = get_child(obj, name=name, _new_name=name, _new_parent=obj)
      if isinstance(mock_children, dict):
        mock_children[name] = child
      return _unbox(child)
    if isinstance(obj, pydantic.BaseModel):
      dumped = obj.model_dump()
      if name in dumped:
        return _unbox(dumped[name])
      extra = obj.model_extra
      if isinstance(extra, dict) and name in extra:
        return _unbox(extra[name])
    if not isinstance(default, _MissingSentinel):
      return _unbox(default)
    raise


def _test_setattr(
  obj: _ObjT,
  name: str,
  value: _ValT,
) -> None:
  """Set attribute ``name`` on ``obj`` via ``unittest.mock.patch.object`` for test setup.

  Args:
      obj: Target object to mutate.
      name: Attribute name to set.
      value: Value to assign.
  """
  p = mock.patch.object(obj, name, value, create=True)
  p.start()


def _test_hasattr(
  obj: _ObjT,
  name: str,
) -> bool:
  """Check whether ``obj`` exposes attribute ``name`` via ``inspect.getattr_static``.

  Args:
      obj: Target object to inspect.
      name: Attribute name to check.

  Returns:
      True if ``obj`` has attribute ``name``, False otherwise.
  """
  try:
    _test_getattr(obj, name)
    return True
  except AttributeError:
    return False
