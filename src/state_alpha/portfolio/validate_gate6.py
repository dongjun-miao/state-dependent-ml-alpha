"""Independent validation of the Gate 6 portfolio artifact."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from state_alpha.config import ProjectConfig, load_yaml
from state_alpha.models.gate4 import _sha256, _snapshot
from state_alpha.models.gate5 import SCORE_COLUMNS
from state_alpha.paths import PathPolicy
from state_alpha.portfolio.core import variant_catalog
from state_alpha.portfolio.gate6 import GATE6_SCHEMA_VERSION


def validate_gate6(
    project_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify hashes, formation integrity, accounting, and source immutability."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    root = config.project_path("data_processed") / "gate6_v1"
    if not root.is_dir():
        raise FileNotFoundError(f"Gate 6 root missing: {root}")
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    contract = load_yaml(config.root / "configs" / "gate6.yaml")
    errors: list[str] = []
    if manifest.get("schema_version") != GATE6_SCHEMA_VERSION:
        errors.append("manifest schema version mismatch")
    if manifest.get("analysis_classification") != (
        "secondary_locked_post_gate5"
    ):
        errors.append("wrong Gate 6 analysis classification")
    if manifest.get(
        "confirmatory_untouched_oos_claim_allowed"
    ) is not False:
        errors.append("Gate 6 incorrectly permits confirmatory claims")
    if manifest.get("strict_oos_used_for_portfolio_selection") is not False:
        errors.append("Gate 6 used OOS for portfolio selection")
    if manifest.get("winner_selection_used") is not False:
        errors.append("Gate 6 selected an OOS winner")
    if manifest.get("formation_domain_uses_future_labels") is not False:
        errors.append("Gate 6 formation domain used future labels")
    if manifest.get("overwrite_used") is not False:
        errors.append("Gate 6 reports an overwrite")

    files_verified = 0
    for item in manifest.get("output_files", []):
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing output: {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch: {item['path']}")
        if _sha256(path) != item["sha256"]:
            errors.append(f"SHA-256 mismatch: {item['path']}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            metadata = parquet.schema_arrow.metadata or {}
            if (
                metadata.get(b"schema_version", b"").decode()
                != GATE6_SCHEMA_VERSION
            ):
                errors.append(f"Parquet schema mismatch: {item['path']}")
            if item["rows"] != parquet.metadata.num_rows:
                errors.append(f"Parquet row mismatch: {item['path']}")
        files_verified += 1

    expected_variants = variant_catalog(contract)
    variants = pd.read_parquet(
        root / "evidence" / "variant_catalog.parquet"
    )
    try:
        pd.testing.assert_frame_equal(
            variants.reset_index(drop=True),
            expected_variants.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError:
        errors.append("Gate 6 variant catalog differs from contract")
    if len(variants) != 100 or variants["signal"].nunique() != 10:
        errors.append("Gate 6 did not carry all 100 portfolio variants")

    formation_rows = 0
    formation_dates: set[int] = set()
    future_columns_found: list[str] = []
    for year in range(2013, 2026):
        path = (
            root
            / "formation_scores"
            / f"year={year}"
            / "part-00000.parquet"
        )
        if not path.is_file():
            errors.append(f"missing formation-score year {year}")
            continue
        frame = pd.read_parquet(path)
        future = [
            column
            for column in frame.columns
            if column.startswith("fwd_") or column.startswith("label_")
        ]
        future_columns_found.extend(future)
        required = {
            "permno",
            "permco",
            "date",
            "session_id",
            "execution_valid",
            "execution_session_id",
            *SCORE_COLUMNS,
        }
        if not required.issubset(frame.columns):
            errors.append(f"formation-score schema incomplete for {year}")
            continue
        if frame.duplicated(["permno", "session_id"]).any():
            errors.append(f"duplicate formation key in {year}")
        if not (
            pd.to_numeric(
                frame["execution_session_id"], errors="coerce"
            )
            == frame["session_id"] + 1
        ).all():
            errors.append(f"wrong execution offset in {year}")
        if not frame["session_id"].mod(5).sub(
            int(contract["formation"]["anchor_session_id"]) % 5
        ).mod(5).eq(0).all():
            errors.append(f"formation cadence changed in {year}")
        for gate in ["linear_capped_gate", "tree_capped_gate"]:
            valid = frame[[gate, "base_family_score"]].dropna()
            if (
                valid[gate].abs()
                > valid["base_family_score"].groupby(
                    frame.loc[valid.index, "date"]
                ).rank(method="average", pct=True).sub(0.5).abs()
                + 1e-6
            ).any():
                errors.append(f"capped gate amplified exposure in {year}")
        formation_rows += len(frame)
        formation_dates.update(frame["session_id"].astype(int).unique())
    if future_columns_found:
        errors.append("future-label columns found in formation scores")
    if formation_rows != manifest.get("formation_score_rows"):
        errors.append("formation-score row total mismatch")
    if len(formation_dates) != manifest.get("formation_dates"):
        errors.append("formation-date count mismatch")

    daily = pd.read_parquet(
        root / "evidence" / "daily_portfolios.parquet"
    )
    if daily.duplicated(["date", "variant_id"]).any():
        errors.append("duplicate daily portfolio key")
    if daily["variant_id"].nunique() != len(variants):
        errors.append("daily portfolio variants are incomplete")
    if len(daily) != manifest.get("daily_portfolio_rows"):
        errors.append("daily portfolio row total mismatch")
    numeric_columns = [
        "gross_return_loss100",
        "gross_return_zero",
        "one_way_turnover",
        *[
            f"net_return_{int(value)}bps"
            for value in contract["turnover_and_costs"][
                "one_way_cost_bps"
            ]
        ],
    ]
    if not np.isfinite(
        daily[numeric_columns].to_numpy(dtype=float)
    ).all():
        errors.append("daily portfolio output contains non-finite values")
    if (daily["one_way_turnover"] < -1e-12).any():
        errors.append("daily one-way turnover is negative")
    for bps in contract["turnover_and_costs"]["one_way_cost_bps"]:
        expected = (
            daily["gross_return_loss100"]
            - 2.0
            * daily["one_way_turnover"]
            * float(bps)
            / 10000.0
        )
        if not np.allclose(
            daily[f"net_return_{int(bps)}bps"],
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            errors.append(f"cost equation failed at {bps} bps")

    cohorts = pd.read_parquet(
        root / "evidence" / "cohort_diagnostics.parquet"
    )
    if cohorts.duplicated(
        ["formation_session_id", "variant_id"]
    ).any():
        errors.append("duplicate cohort diagnostic key")
    expected_cohorts = int(manifest.get("formation_dates", 0)) * len(
        variants
    )
    if len(cohorts) != expected_cohorts:
        errors.append("cohort diagnostic coverage is incomplete")
    if len(cohorts) != manifest.get("cohort_diagnostic_rows"):
        errors.append("cohort row total differs from manifest")
    if cohorts["prefill_net_weight"].abs().max() > 1e-8:
        errors.append("prefill portfolio is not self-financing")
    if not np.allclose(
        cohorts["prefill_gross_weight"], 1.0, atol=1e-8
    ):
        errors.append("prefill portfolio gross is not one")
    sleeve = float(
        contract["formation"]["fixed_capital_fraction_per_cohort"]
    )
    if not np.allclose(
        cohorts["cash_weight_for_self_financing"]
        + sleeve * cohorts["postfill_net_weight"],
        0.0,
        atol=1e-12,
    ):
        errors.append("postfill cash accounting is not self-financing")
    neutral = cohorts["variant_id"].str.contains(
        "__risk_neutralized__"
    )
    tolerance = float(
        contract["neutralization"]["target_tolerance"]
    )
    if (
        cohorts.loc[
            neutral, "prefill_max_abs_constraint"
        ].max()
        > tolerance * 1.01
    ):
        errors.append("neutralized prefill constraints exceed tolerance")
    if (
        cohorts["missing_bound_gross_weight"]
        > cohorts["postfill_gross_weight"] + 1e-12
    ).any():
        errors.append("missing-return notional exceeds filled gross")

    source_roots = {
        "gate4": config.project_path("data_processed") / "gate4_v1",
        "gate5": config.project_path("data_processed") / "gate5_v1",
        "features": config.project_path("data_interim") / "features_v1",
        "phase2b": config.project_path("data_interim") / "phase2b_v1",
        "canonical": config.project_path("data_interim") / "canonical_v1",
    }
    source_checks: dict[str, bool] = {}
    for name, path in source_roots.items():
        passed = _snapshot(path) == manifest["source_snapshots"][name]
        source_checks[f"{name}_input_unchanged"] = passed
        if not passed:
            errors.append(f"{name} source artifact changed")
    config_unchanged = (
        _sha256(config.root / "configs" / "gate6.yaml")
        == manifest.get("gate6_config_sha256")
    )
    if not config_unchanged:
        errors.append("Gate 6 config changed after build")

    result = {
        "schema_version": GATE6_SCHEMA_VERSION,
        "validated_at": datetime.now().isoformat(),
        "gate6_root": str(root),
        "passed": not errors,
        "errors": errors,
        "output_files_verified": files_verified,
        "formation_dates": len(formation_dates),
        "formation_score_rows": formation_rows,
        "portfolio_variants": len(variants),
        "daily_portfolio_rows": len(daily),
        "cohort_diagnostic_rows": len(cohorts),
        "future_label_columns_in_formation_scores": sorted(
            set(future_columns_found)
        ),
        "cost_equations_verified": len(
            contract["turnover_and_costs"]["one_way_cost_bps"]
        ),
        "neutralized_constraint_max": float(
            cohorts.loc[
                neutral, "prefill_max_abs_constraint"
            ].max()
        ),
        "gate6_config_unchanged": config_unchanged,
        "strict_oos_used_for_portfolio_selection": manifest.get(
            "strict_oos_used_for_portfolio_selection"
        ),
        "winner_selection_used": manifest.get("winner_selection_used"),
        "formation_domain_uses_future_labels": manifest.get(
            "formation_domain_uses_future_labels"
        ),
        "analysis_classification": manifest.get(
            "analysis_classification"
        ),
        "confirmatory_untouched_oos_claim_allowed": manifest.get(
            "confirmatory_untouched_oos_claim_allowed"
        ),
        **source_checks,
        "overwrite_used": manifest.get("overwrite_used"),
    }
    report_path = (
        output_path
        if output_path is not None
        else config.project_path("reports_tables")
        / "gate6_validation.json"
    )
    report_path = policy.assert_writable(report_path)
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Gate 6 validation: {report_path}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        raise ValueError(f"Gate 6 validation failed: {errors[:8]}")
    return result
