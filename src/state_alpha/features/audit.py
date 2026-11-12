"""Development-only audit of the Gate 3 price representation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_alpha.config import ProjectConfig
from state_alpha.paths import PathPolicy


AUDIT_COLUMNS = [
    "permno",
    "date",
    "price_raw",
    "previous_price_raw",
    "price_factor",
    "return_total",
    "return_ex_distribution",
    "bid_raw",
    "ask_raw",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
]


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    valid = frame["primary_rows"].gt(0) & frame[column].notna()
    return float(
        np.average(
            frame.loc[valid, column],
            weights=frame.loc[valid, "primary_rows"],
        )
    )


def audit_price_representation(project_root: Path) -> dict[str, Path]:
    """Audit price/return fields using the frozen 2000-2012 universe only."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    canonical = config.project_path("data_interim") / "canonical_v1" / "security_day"
    phase2b = config.project_path("data_interim") / "phase2b_v1" / "panel"
    if not canonical.is_dir() or not phase2b.is_dir():
        raise FileNotFoundError("Canonical and Phase 2B inputs must exist")
    table_dir = policy.ensure_directory(config.project_path("reports_tables"))
    memo_dir = policy.ensure_directory(config.project_path("reports_memo"))
    outputs = {
        "annual": table_dir / "phase3_price_representation_audit.csv",
        "memo": memo_dir / "PHASE3_PRICE_REPRESENTATION_AUDIT.md",
    }
    for path in outputs.values():
        resolved = policy.assert_writable(path)
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite audit output: {resolved}")

    records: list[dict[str, Any]] = []
    for year in range(2000, 2013):
        panel = pd.read_parquet(
            phase2b / f"year={year}",
            columns=["permno", "date", "universe_primary"],
        )
        keys = panel.loc[
            panel["universe_primary"], ["permno", "date"]
        ].copy()
        keys["date"] = pd.to_datetime(keys["date"], errors="raise")
        security = pd.read_parquet(
            canonical / f"year={year}", columns=AUDIT_COLUMNS
        )
        security["date"] = pd.to_datetime(
            security["date"], errors="raise"
        )
        joined = keys.merge(
            security,
            on=["permno", "date"],
            how="left",
            validate="one_to_one",
        )
        price = pd.to_numeric(joined["price_raw"], errors="coerce").abs()
        previous = pd.to_numeric(
            joined["previous_price_raw"], errors="coerce"
        ).abs()
        factor = pd.to_numeric(joined["price_factor"], errors="coerce")
        implied = price * factor / previous - 1.0
        residual = (
            implied
            - pd.to_numeric(
                joined["return_ex_distribution"], errors="coerce"
            )
        ).abs()
        bid = pd.to_numeric(joined["bid_raw"], errors="coerce")
        ask = pd.to_numeric(joined["ask_raw"], errors="coerce")
        midpoint = (bid + ask) / 2.0
        valid_quote = bid.gt(0) & ask.ge(bid) & midpoint.gt(0)
        complete_ohlc = joined[
            ["open_raw", "high_raw", "low_raw", "close_raw"]
        ].notna().all(axis=1)
        records.append(
            {
                "year": year,
                "primary_rows": len(joined),
                "return_total_coverage": joined["return_total"].notna().mean(),
                "return_ex_distribution_coverage": joined[
                    "return_ex_distribution"
                ].notna().mean(),
                "price_factor_coverage": factor.notna().mean(),
                "nonunit_price_factor_rows": int(
                    (factor.notna() & ~np.isclose(factor, 1.0)).sum()
                ),
                "price_identity_pairs": int(residual.notna().sum()),
                "price_identity_abs_residual_p50": residual.median(),
                "price_identity_abs_residual_p99": residual.quantile(0.99),
                "price_identity_abs_residual_max": residual.max(),
                "price_identity_outliers_gt_1e_4": int(
                    residual.gt(1e-4).sum()
                ),
                "valid_quoted_spread_coverage": valid_quote.mean(),
                "complete_raw_ohlc_coverage": complete_ohlc.mean(),
            }
        )
    annual = pd.DataFrame(records)
    annual.to_csv(outputs["annual"], index=False)

    positive = annual.loc[annual["primary_rows"].gt(0)]
    total_rows = int(positive["primary_rows"].sum())
    nonunit = int(positive["nonunit_price_factor_rows"].sum())
    outliers = int(positive["price_identity_outliers_gt_1e_4"].sum())
    retx_coverage = _weighted_mean(
        positive, "return_ex_distribution_coverage"
    )
    spread_coverage = _weighted_mean(
        positive, "valid_quoted_spread_coverage"
    )
    ohlc_coverage = _weighted_mean(
        positive, "complete_raw_ohlc_coverage"
    )
    outputs["memo"].write_text(
        f"""# Phase 3 price-representation audit

Generated: {datetime.now().isoformat(timespec="seconds")}

## Information boundary

This audit uses only the frozen primary universe in 2000-2012. It examines
field coverage and accounting consistency only. It does not inspect labels,
IC, model fit, portfolio returns, or any 2013-2025 outcome.

## Result

- Audited primary security-days: {total_rows:,}.
- `DlyRetx` coverage: {retx_coverage:.6%}.
- Non-unit `DlyFacPrc` rows: {nonunit:,}.
- Raw price/factor identity residuals above `1e-4`: {outliers:,}. These rare
  discontinuities make reconstruction from adjacent raw prices an inferior
  primary contract even though the 99th-percentile residual is near source
  rounding precision.
- Valid positive bid/ask coverage: {spread_coverage:.6%}; missingness is
  retained, especially in early years.
- Complete raw OHLC coverage: {ohlc_coverage:.6%}.

## Frozen decision

All rolling price-path features use compounded `DlyRetx`, CRSP's daily price
return without dividends. Raw OHLC is not used in the primary Gate 3 price
chain because its levels require a separate adjustment/continuity contract.
Absolute current price remains valid for the price-level and dollar-volume
states, while bid/ask is used only when the quote is positive and ordered.

## Authoritative references

- CRSP Cross Reference Guide:
  https://www.crsp.org/wp-content/uploads/guides/CRSP_Cross_Reference_Guide_1.0_to_2.0.pdf
- CRSP Calculations and Index Methodologies:
  https://www.crsp.org/wp-content/uploads/guides/CRSP_Calculations_and_Index_Methodologies.pdf
""",
        encoding="utf-8",
    )
    return outputs
