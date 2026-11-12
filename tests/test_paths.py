from __future__ import annotations

from pathlib import Path

import pytest

from state_alpha.paths import PathPolicy, PathPolicyError


def test_nested_project_allows_only_owned_runtime_subtrees(tmp_path: Path) -> None:
    project_1 = tmp_path / "project_1"
    project_2 = project_1 / "project_2"
    raw_root = tmp_path / "raw"
    allowed = project_2 / "data" / "interim"
    allowed.mkdir(parents=True)
    raw_root.mkdir()
    policy = PathPolicy(
        project_root=project_2.resolve(),
        allowed_write_roots=(allowed.resolve(),),
        raw_root=raw_root.resolve(),
        project_1_root=project_1.resolve(),
    )

    assert policy.assert_writable(allowed / "canonical") == (
        allowed / "canonical"
    ).resolve()
    with pytest.raises(PathPolicyError):
        policy.assert_writable(project_1 / "outputs")
    with pytest.raises(PathPolicyError):
        policy.assert_writable(project_2 / "README.md")
    with pytest.raises(PathPolicyError):
        policy.assert_writable(raw_root / "raw.csv")


def test_read_source_must_exist_under_raw_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    project_root = tmp_path / "project"
    source = raw_root / "source.zip"
    raw_root.mkdir()
    project_root.mkdir()
    source.write_bytes(b"test")
    policy = PathPolicy(
        project_root=project_root.resolve(),
        allowed_write_roots=(project_root.resolve(),),
        raw_root=raw_root.resolve(),
        project_1_root=tmp_path.resolve(),
    )

    assert policy.assert_read_only_source(source) == source.resolve()
    with pytest.raises(PathPolicyError):
        policy.assert_read_only_source(project_root)

