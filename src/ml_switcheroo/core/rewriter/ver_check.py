"""
Versioning Check Mixin.

Handles parsing and verification of framework version strings against constraints
defined in the Semantic Knowledge Base. This ensures that generated code does
not rely on APIs present only in future or past versions of the target framework.
"""

from typing import Optional, Tuple
import importlib.metadata
import re


class VersioningMixin:
  """
  Mixin for checking target framework version compatibility.

  Provides methods to load the current version (from config or environment)
  and compare it against min/max constraints defined in ODL.

  Assumes it is mixed into a class providing `target_fw` and `semantics`.
  """

  def __init__(self, *args, **kwargs):
    """
    Initialize cache state.

    Accepts *args, **kwargs to support cooperative multiple inheritance chains
    where `super().__init__` propagates arguments (like `context`).
    """
    self._cached_target_version: Optional[str] = None
    self._version_checked = False
    super().__init__(*args, **kwargs)

  def _get_target_version(self) -> Optional[str]:
    """
    Resolves the version of the target framework.
    Prioritizes configuration overrides (e.g. Ghost snapshot metadata),
    then falls back to installed package metadata.

    Returns:
        str: The version string (e.g. "1.12.0") or None if unknown.
    """
    if self._version_checked:
      return self._cached_target_version

    self._version_checked = True

    # 1. Check if Semantic Config has a 'version' set (e.g. from Ghost Snapshot header)
    fw_conf = self.semantics.get_framework_config(self.target_fw)
    if fw_conf and "version" in fw_conf:
      self._cached_target_version = fw_conf["version"]
      return self._cached_target_version

    # 2. Try importing live metadata
    # Map special package names (e.g. flax_nnx -> flax)
    pkg = self.target_fw
    if pkg == "flax_nnx":
      pkg = "flax"

    try:
      self._cached_target_version = importlib.metadata.version(pkg)
    except Exception:
      self._cached_target_version = None

    return self._cached_target_version

  def _parse_version(self, v_str: str) -> Tuple[int, ...]:
    """
    Parses version string into tuple of ints for comparison.
    Handles basic semver like '1.2.3' or versions with build metadata '2.0.0+cuda'.

    Args:
        v_str: Version string.

    Returns:
        Tuple of integers (e.g. (1, 2, 3)).
    """
    parts = []
    # Split by non-digits to handle '.', '+', '-', etc.
    tokens = re.split(r"[^\d]+", v_str)
    for t in tokens:
      if t:
        parts.append(int(t))
    return tuple(parts)

  def check_version_constraints(self, min_v: Optional[str], max_v: Optional[str]) -> Optional[str]:
    """
    Verifies loaded target version against provided constraints.

    Args:
        min_v: Minimum required version string (inclusive).
        max_v: Maximum supported version string (exclusive/warning threshold).

    Returns:
        None if compatible.
        String warning message if incompatible.
    """
    if not min_v and not max_v:
      return None

    current = self._get_target_version()
    if not current:
      # Can't verify if framework not installed/detected
      return None

    curr_tuple = self._parse_version(current)

    if min_v:
      min_tuple = self._parse_version(min_v)
      if curr_tuple < min_tuple:
        return f"Target {self.target_fw}@{current} is older than required {min_v}"

    if max_v:
      max_tuple = self._parse_version(max_v)
      # Treat max_version as loose upper bound (deprecated after this)
      if curr_tuple >= max_tuple:
        return f"Target {self.target_fw}@{current} exceeds max supported {max_v}"

    return None
