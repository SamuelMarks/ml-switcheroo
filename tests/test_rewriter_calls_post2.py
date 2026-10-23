"""Test suite for rewriter calls post2."""

import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class DummyTraits:
  """Test helper class DummyTraits."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.strip_magic_args: list[str] = []
    self.auto_strip_magic_args: bool = False
    self.inject_magic_args: list[tuple[str, str]] = []


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self, is_init: bool = False, is_module: bool = False) -> None:
    """Initialize test helper instance.

    Args:
        is_init: The is init (bool) for   init  .
        is_module: The is module (bool) for   init  .
    """

    class Sig:
      """Test helper class Sig."""

      def __init__(self, i: bool, m: bool) -> None:
        """Initialize test helper instance.

        Args:
            i: The i (bool) for   init  .
            m: The m (bool) for   init  .
        """
        self.is_init = i
        self.is_module_method = m

    self.signature_stack: list[Sig] = [Sig(is_init, is_module)] if is_init or is_module else []


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self, origins: dict[str, str] | None = None, magic: list[str] | None = None) -> None:
    """Initialize test helper instance.

    Args:
        origins: The origins (dict[str, str] | None) for   init  .
        magic: The magic (list[str] | None) for   init  .
    """
    self._key_origins: dict[str, str] = origins or {}
    if magic is not None:
      self.known_magic_args: list[str] = magic


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(
    self,
    context: DummyContext | None = None,
    semantics: DummySemantics | None = None,
    traits: DummyTraits | None = None,
    report: bool = True,
  ) -> None:
    """Initialize test helper instance.

    Args:
        context: Execution or rewriter context instance.
        semantics: SemanticsManager instance providing operation and trait definitions.
        traits: The traits (DummyTraits | None) for   init  .
        report: The report (bool) for   init  .
    """
    self.context: DummyContext = context or DummyContext()
    self.semantics: DummySemantics = semantics or DummySemantics()
    self.traits: DummyTraits = traits or DummyTraits()
    self.report: bool = report

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Execute create dotted name helper logic.

    Returns:
        Resulting `cst.Name` from create dotted name.
    """
    if name == "error":
      raise ValueError("bad type")
    return cst.Name(name)

  def _report_failure(self, msg: str) -> None:
    """Execute report failure helper logic.

    Args:
        msg: Warning or error message string.
    """
    pass

  def _get_target_traits(self) -> DummyTraits:
    """Execute get target traits helper logic.

    Returns:
        Resulting `DummyTraits` from get target traits.
    """
    return self.traits


def test_post_branches() -> None:
  """Verify post branches behavior."""
  rewriter: DummyRewriter = DummyRewriter()
  node: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")

  # 38 -> 47 (missing output_select_index)
  handle_post_processing(rewriter, node, {}, "id")

  # 47 -> 59 (missing output_cast)
  handle_post_processing(rewriter, node, {"output_select_index": None}, "id")

  # 38 -> 39, 47 -> 48 -> 59
  mapping = {"output_select_index": 0, "output_cast": "float32"}
  node_tuple: cst.Subscript = getattr(getattr(cst.parse_statement("f()[0]"), "body")[0], "value")
  handle_post_processing(rewriter, node_tuple, mapping, "id")

  # 59 -> 99 (no context signature stack)
  _res = handle_post_processing(rewriter, node, mapping, "id")
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(_res)])]).code.strip() == "f()[0].astype(float32)"
