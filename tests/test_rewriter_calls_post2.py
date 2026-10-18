"""Test suite for rewriter calls post2."""

from typing import Any, Dict, List, Optional, Tuple

import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class DummyTraits:
  """Test helper class DummyTraits."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.strip_magic_args: List[str] = []
    self.auto_strip_magic_args: bool = False
    self.inject_magic_args: List[Tuple[str, str]] = []


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self, is_init: bool = False, is_module: bool = False) -> None:
    """Initialize test helper instance.

    Args:
        is_init: Test argument or fixture.
        is_module: Test argument or fixture.
    """

    class Sig:
      """Test helper class Sig."""

      def __init__(self, i: bool, m: bool) -> None:
        """Initialize test helper instance.

        Args:
            i: Test argument or fixture.
            m: Test argument or fixture.
        """
        self.is_init = i
        self.is_module_method = m

    self.signature_stack: List[Sig] = [Sig(is_init, is_module)] if is_init or is_module else []


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self, origins: Optional[Dict[str, str]] = None, magic: Optional[List[str]] = None) -> None:
    """Initialize test helper instance.

    Args:
        origins: Test argument or fixture.
        magic: Test argument or fixture.
    """
    self._key_origins: Dict[str, str] = origins or {}
    if magic is not None:
      self.known_magic_args: List[str] = magic


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(
    self,
    context: Optional[DummyContext] = None,
    semantics: Optional[DummySemantics] = None,
    traits: Optional[DummyTraits] = None,
    report: bool = True,
  ) -> None:
    """Initialize test helper instance.

    Args:
        context: Test argument or fixture.
        semantics: Test argument or fixture.
        traits: Test argument or fixture.
        report: Test argument or fixture.
    """
    self.context: DummyContext = context or DummyContext()
    self.semantics: DummySemantics = semantics or DummySemantics()
    self.traits: DummyTraits = traits or DummyTraits()
    self.report: bool = report

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Execute create dotted name helper logic.

    Returns:
        Computed test result or fixture value.
    """
    if name == "error":
      raise ValueError("bad type")
    return cst.Name(name)

  def _report_failure(self, msg: str) -> None:
    """Execute report failure helper logic.

    Args:
        msg: Test argument or fixture.
    """
    pass

  def _get_target_traits(self) -> DummyTraits:
    """Execute get target traits helper logic.

    Returns:
        Computed test result or fixture value.
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
  mapping: Dict[str, Any] = {"output_select_index": 0, "output_cast": "float32"}
  node_tuple: cst.Subscript = getattr(getattr(cst.parse_statement("f()[0]"), "body")[0], "value")
  handle_post_processing(rewriter, node_tuple, mapping, "id")

  # 59 -> 99 (no context signature stack)
  _res = handle_post_processing(rewriter, node, mapping, "id")
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(_res)])]).code.strip() == "f()[0].astype(float32)"
