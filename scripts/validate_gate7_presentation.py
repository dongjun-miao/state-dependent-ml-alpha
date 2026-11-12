"""Independently validate the Gate 7 presentation package."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/gate7.yaml"))
    return parser.parse_args()


def validate(project_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(project_root / "src"))
    from state_alpha.reporting.gate7 import (  # noqa: PLC0415
        FIGURE_FILES,
        sha256_file,
        verify_locked_sources,
    )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checks: dict[str, Any] = {}
    failures: list[str] = []

    try:
        locked_hashes = verify_locked_sources(project_root, config)
        checks["locked_sources"] = {"passed": True, "count": len(locked_hashes)}
    except Exception as exc:  # validation must retain all observed failures
        locked_hashes = {}
        checks["locked_sources"] = {"passed": False, "error": str(exc)}
        failures.append(str(exc))

    gate6_validation_path = project_root / "reports/tables/gate6_validation.json"
    gate6_validation = json.loads(gate6_validation_path.read_text(encoding="utf-8"))
    gate6_passed = gate6_validation.get("passed") is True
    checks["gate6_validation_passed"] = gate6_passed
    if not gate6_passed:
        failures.append("Gate 6 validation is not passed=true")

    figure_dir = output_root / config["figures"]["output_directory"]
    figure_manifest_path = figure_dir / "gate7_figure_manifest.json"
    if not figure_manifest_path.is_file():
        failures.append(f"Missing figure manifest: {figure_manifest_path}")
        figure_manifest: dict[str, Any] = {}
    else:
        figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    figure_checks: dict[str, Any] = {}
    for name in FIGURE_FILES:
        path = figure_dir / name
        item: dict[str, Any] = {"exists": path.is_file()}
        if path.is_file():
            with Image.open(path) as image:
                item["width_px"], item["height_px"] = image.size
            item["sha256"] = sha256_file(path)
            expected = figure_manifest.get("figures", {}).get(name)
            item["manifest_hash_match"] = expected == item["sha256"]
            item["resolution_passed"] = item["width_px"] >= 1600 and item["height_px"] >= 900
            if not item["manifest_hash_match"]:
                failures.append(f"Figure manifest hash mismatch: {name}")
            if not item["resolution_passed"]:
                failures.append(f"Figure resolution too small: {name}")
        else:
            failures.append(f"Missing figure: {name}")
        figure_checks[name] = item
    checks["figures"] = figure_checks
    checks["figure_count"] = len([name for name in FIGURE_FILES if (figure_dir / name).is_file()])
    if checks["figure_count"] != 7:
        failures.append(f"Expected 7 figures, found {checks['figure_count']}")

    config_hash = sha256_file(config_path)
    if figure_manifest.get("config_sha256") != config_hash:
        failures.append("Figure manifest config hash mismatch")
    if locked_hashes and figure_manifest.get("locked_source_hashes") != locked_hashes:
        failures.append("Figure manifest locked-source hashes do not match current verified sources")

    pdf_path = output_root / config["report"]["output_pdf"]
    pdf_check: dict[str, Any] = {"exists": pdf_path.is_file()}
    if pdf_path.is_file():
        reader = PdfReader(str(pdf_path))
        page_text = [(page.extract_text() or "") for page in reader.pages]
        full_text = "\n".join(page_text)
        pdf_check.update(
            {
                "pages": len(reader.pages),
                "sha256": sha256_file(pdf_path),
                "minimum_text_chars_per_page": min(len(text.strip()) for text in page_text),
                "replacement_character_count": full_text.count("\ufffd"),
            }
        )
        if not (10 <= len(reader.pages) <= 16):
            failures.append(f"Unexpected PDF page count: {len(reader.pages)}")
        if pdf_check["minimum_text_chars_per_page"] < 80:
            failures.append("At least one PDF page has too little extractable text")
        if pdf_check["replacement_character_count"]:
            failures.append("PDF text contains replacement characters")
        required_text = [
            "Learning When Technical Signals Work",
            "presentation-only",
            "future label availability",
            "No OOS winner",
            "cost and capacity",
            "Reference-NAV ADV participation",
            "Reproducibility and provenance",
        ]
        missing_text = [phrase for phrase in required_text if phrase not in full_text]
        pdf_check["required_text_missing"] = missing_text
        if missing_text:
            failures.append("PDF is missing required text: " + ", ".join(missing_text))
    else:
        failures.append(f"Missing PDF: {pdf_path}")
    checks["pdf"] = pdf_check

    doc_paths = {
        "interview_brief": output_root / config["report"]["interview_brief"],
        "reproducibility_sheet": output_root / config["report"]["reproducibility_sheet"],
    }
    doc_checks: dict[str, Any] = {}
    for name, path in doc_paths.items():
        item = {"exists": path.is_file()}
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            item.update({"sha256": sha256_file(path), "characters": len(text)})
            if len(text) < 1000:
                failures.append(f"Presentation document is unexpectedly short: {path}")
        else:
            failures.append(f"Missing presentation document: {path}")
        doc_checks[name] = item
    checks["documents"] = doc_checks

    output_files = [pdf_path, figure_manifest_path, *[figure_dir / name for name in FIGURE_FILES], *doc_paths.values()]
    output_hashes = {
        str(path.relative_to(output_root)).replace("\\", "/"): sha256_file(path)
        for path in output_files
        if path.is_file()
    }
    audit_manifest = {
        "schema": "state_alpha_gate7_audit_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "presentation_only": True,
        "config_sha256": config_hash,
        "locked_source_hashes": locked_hashes,
        "output_hashes": output_hashes,
        "figure_count": checks["figure_count"],
        "pdf_pages": pdf_check.get("pages"),
    }
    audit_path = output_root / config["report"]["audit_manifest"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_manifest, indent=2, sort_keys=True), encoding="utf-8")

    result = {
        "schema": "state_alpha_gate7_validation_v1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "presentation_only": True,
        "config_sha256": config_hash,
        "checks": checks,
        "audit_manifest": str(audit_path.relative_to(output_root)).replace("\\", "/"),
        "audit_manifest_sha256": sha256_file(audit_path),
        "failures": failures,
    }
    validation_path = output_root / config["report"]["validation_report"]
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or project_root).resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    result = validate(project_root, output_root, config_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
