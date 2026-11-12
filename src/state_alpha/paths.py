"""Runtime path containment and write-safety controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from state_alpha.config import ConfigurationError, ProjectConfig


class PathPolicyError(PermissionError):
    """Raised when a runtime write target violates project containment."""


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class PathPolicy:
    """Allow writes only within explicitly configured Project 2 subtrees."""

    project_root: Path
    allowed_write_roots: tuple[Path, ...]
    raw_root: Path
    project_1_root: Path

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "PathPolicy":
        try:
            relative_roots = config.paths["path_policy"][
                "allowed_runtime_write_roots"
            ]
        except KeyError as exc:
            raise ConfigurationError(
                "paths.yaml must define path_policy.allowed_runtime_write_roots"
            ) from exc
        if not isinstance(relative_roots, list) or not relative_roots:
            raise ConfigurationError("allowed_runtime_write_roots must be a list")
        project_root = config.root.resolve()
        allowed = tuple((project_root / value).resolve() for value in relative_roots)
        if any(not _is_relative_to(path, project_root) for path in allowed):
            raise ConfigurationError("Configured write root escapes Project 2")
        return cls(
            project_root=project_root,
            allowed_write_roots=allowed,
            raw_root=config.external_path("data_root"),
            project_1_root=config.external_path("project_1_root"),
        )

    def assert_read_only_source(self, path: Path) -> Path:
        """Resolve a source and require it to exist under shared raw storage."""

        resolved = path.resolve(strict=True)
        if not _is_relative_to(resolved, self.raw_root):
            raise PathPolicyError(f"Source is outside shared raw storage: {resolved}")
        return resolved

    def assert_writable(self, path: Path) -> Path:
        """Resolve a candidate and require an explicitly allowed write root."""

        resolved = path.resolve(strict=False)
        if not _is_relative_to(resolved, self.project_root):
            raise PathPolicyError(f"Write target escapes Project 2: {resolved}")
        if not any(
            resolved == root or _is_relative_to(resolved, root)
            for root in self.allowed_write_roots
        ):
            raise PathPolicyError(
                f"Runtime write target is not in an allowed Project 2 subtree: "
                f"{resolved}"
            )
        return resolved

    def ensure_directory(self, path: Path) -> Path:
        resolved = self.assert_writable(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

