"""Configuration loading and project-root discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_MARKERS = ("pyproject.toml", "configs/paths.yaml")


class ConfigurationError(RuntimeError):
    """Raised when a project configuration is missing or malformed."""


def discover_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing the project markers."""

    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if all((candidate / marker).is_file() for marker in PROJECT_MARKERS):
            return candidate
    raise ConfigurationError(
        f"Could not find project root from {current}; expected {PROJECT_MARKERS}."
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping documents."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Expected YAML mapping in {resolved}")
    return value


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration bundle."""

    root: Path
    paths: dict[str, Any]
    data_contract: dict[str, Any]

    @classmethod
    def load(cls, project_root: Path | None = None) -> "ProjectConfig":
        root = (
            project_root.resolve()
            if project_root is not None
            else discover_project_root()
        )
        paths = load_yaml(root / "configs" / "paths.yaml")
        contract = load_yaml(root / "configs" / "data_contract.yaml")
        return cls(root=root, paths=paths, data_contract=contract)

    def external_path(self, key: str) -> Path:
        try:
            value = self.paths["external"][key]
        except KeyError as exc:
            raise ConfigurationError(f"Missing external path key: {key}") from exc
        return Path(value).expanduser().resolve()

    def project_path(self, key: str) -> Path:
        try:
            value = self.paths["project"][key]
        except KeyError as exc:
            raise ConfigurationError(f"Missing project path key: {key}") from exc
        if key == "root":
            return self.root
        return (self.root / value).resolve()

