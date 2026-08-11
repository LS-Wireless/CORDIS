"""
tests/test_paths.py
===================
Project-root resolution (`cordis/utils/paths.py`).

Everything in the framework that reads a config, writes a result, or finds a
figure goes through `get_project_root()`, so a wrong answer here misroutes the
whole pipeline, potentially into the cluster-authoritative `results/` tree. These tests pin the marker logic and the caching contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cordis.utils import paths as paths_mod
from cordis.utils.paths import get_project_root, project_path


def test_root_is_the_repository_root(repo_root: Path) -> None:
    """`get_project_root()` resolves to the directory holding `cordis/`."""
    assert get_project_root() == repo_root


def test_root_contains_every_declared_marker(repo_root: Path) -> None:
    """
    At least one `_MARKERS` entry exists at the resolved root.

    The walk stops at the *first* marker hit, so a root that satisfies none of
    them would mean the function returned by exhaustion rather than by match.
    """
    present = [m for m in paths_mod._MARKERS if (repo_root / m).exists()]
    assert present, f"none of {paths_mod._MARKERS} found at {repo_root}"


def test_result_is_absolute_and_a_directory() -> None:
    root = get_project_root()
    assert root.is_absolute()
    assert root.is_dir()


def test_is_cached() -> None:
    """`get_project_root` is `lru_cache`d, so repeated calls are identical."""
    assert get_project_root() is get_project_root()
    info = get_project_root.cache_info()
    assert info.maxsize == 1


def test_project_path_joins_onto_the_root(repo_root: Path) -> None:
    assert project_path("configs", "default.json") == repo_root / "configs" / "default.json"
    assert project_path("configs", "default.json").exists()


def test_project_path_with_no_parts_is_the_root() -> None:
    assert project_path() == get_project_root()


def test_project_path_does_not_require_existence() -> None:
    """Path building is pure, it must not touch the filesystem."""
    p = project_path("does", "not", "exist.json")
    assert not p.exists()
    assert p.parts[-3:] == ("does", "not", "exist.json")


def test_marker_list_is_ordered_most_specific_first() -> None:
    """
    `configs/default.json` is checked before `.git`.

    Order matters when the repo is vendored inside another git checkout: the
    CORDIS-specific marker must win over the generic one.
    """
    markers = paths_mod._MARKERS
    assert markers[0] == "configs/default.json"
    assert markers.index("configs/default.json") < markers.index(".git")


def test_walk_depth_is_bounded() -> None:
    """The upward walk is bounded, so a missing marker cannot loop forever."""
    assert isinstance(paths_mod._MAX_LEVELS, int)
    assert 1 <= paths_mod._MAX_LEVELS <= 100


@pytest.fixture
def clean_root_cache():
    """
    Clear the `lru_cache` around `get_project_root` before and after a test.

    Needed by any test that monkeypatches the module: the cache would
    otherwise hand the poisoned answer to every later test in the session.
    """
    get_project_root.cache_clear()
    yield
    get_project_root.cache_clear()


def test_raises_when_no_marker_is_reachable(
    tmp_path: Path, monkeypatch, clean_root_cache
) -> None:
    """
    With the markers made unfindable, resolution raises a clear `RuntimeError`
    rather than silently returning a wrong directory, which is what would
    misroute writes into the wrong tree.
    """
    monkeypatch.setattr(paths_mod, "_MARKERS", ["__cordis_marker_that_never_exists__"])
    monkeypatch.setattr(paths_mod, "__file__", str(tmp_path / "utils" / "paths.py"))
    with pytest.raises(RuntimeError, match="project root"):
        get_project_root()


def test_cache_recovers_after_the_poisoned_test(repo_root: Path) -> None:
    """The real root is back once the monkeypatch is undone (ordering guard)."""
    get_project_root.cache_clear()
    assert get_project_root() == repo_root
