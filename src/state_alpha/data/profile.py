"""Descriptive Phase 2A universe and unit profiling from canonical Parquet."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from state_alpha.config import ProjectConfig
from state_alpha.logging_utils import configure_run_logger
from state_alpha.paths import PathPolicy


PROFILE_COLUMNS = [
    "permno",
    "date",
    "primary_exchange",
    "trading_status",
    "security_type",
    "security_subtype",
    "share_type",
    "us_incorporated_flag",
    "issuer_type",
    "daily_delisting_flag",
    "price_raw",
    "market_cap_source",
    "return_total",
    "volume",
    "shares_outstanding_source",
]

QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _refuse_existing(paths: list[Path], policy: PathPolicy) -> None:
    for path in paths:
        resolved = policy.assert_writable(path)
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")


def _quantile_records(
    year: int,
    name: str,
    values: pd.Series,
    *,
    development_period: bool,
) -> list[dict[str, Any]]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    quantiles = clean.quantile(QUANTILES)
    return [
        {
            "year": year,
            "development_period": development_period,
            "variable": name,
            "quantile": float(quantile),
            "value": float(value),
            "nonmissing_rows": int(len(clean)),
        }
        for quantile, value in quantiles.items()
    ]


def _weighted_rate(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    denom = frame[denominator].sum()
    return float(frame[numerator].sum() / denom) if denom else float("nan")


def profile_universe(project_root: Path) -> dict[str, Path]:
    """Create descriptive annual tables without selecting a final universe."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    canonical_root = config.project_path("data_interim") / "canonical_v1"
    security_root = canonical_root / "security_day"
    if not security_root.is_dir():
        raise FileNotFoundError(f"Canonical security-day dataset not found: {security_root}")

    tables_dir = policy.ensure_directory(config.project_path("reports_tables"))
    memo_dir = policy.ensure_directory(config.project_path("reports_memo"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_run_logger(
        "state_alpha.phase2a.profile",
        config.project_path("logs") / f"phase2a_profile_{timestamp}.log",
        policy,
    )
    outputs = {
        "annual": tables_dir / "phase2a_universe_annual_summary.csv",
        "quantiles": tables_dir / "phase2a_universe_quantiles.csv",
        "units": tables_dir / "phase2a_source_unit_diagnostics.csv",
        "memo": memo_dir / "PHASE2A_DATA_FOUNDATION.md",
    }
    _refuse_existing(list(outputs.values()), policy)

    annual_records: list[dict[str, Any]] = []
    quantile_records: list[dict[str, Any]] = []
    unit_records: list[dict[str, Any]] = []
    year_dirs = sorted(security_root.glob("year=*"))
    if not year_dirs:
        raise FileNotFoundError(f"No year partitions under {security_root}")

    for year_dir in year_dirs:
        year = int(year_dir.name.split("=", 1)[1])
        logger.info("Profiling year %d", year)
        frame = ds.dataset(year_dir, format="parquet").to_table(
            columns=PROFILE_COLUMNS
        ).to_pandas()
        price = pd.to_numeric(frame["price_raw"], errors="coerce").abs()
        market_cap_source = pd.to_numeric(
            frame["market_cap_source"], errors="coerce"
        )
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        shares_source = pd.to_numeric(
            frame["shares_outstanding_source"], errors="coerce"
        )
        base = (
            frame["security_type"].eq("EQTY")
            & frame["security_subtype"].eq("COM")
            & frame["share_type"].eq("NS")
            & frame["trading_status"].eq("A")
            & frame["primary_exchange"].isin(["N", "Q", "A"])
        )
        us_incorporated = base & frame["us_incorporated_flag"].eq("Y")
        valid_price = price.gt(0)
        valid_cap = market_cap_source.gt(0)
        valid_volume = volume.ge(0)
        valid_shares = shares_source.gt(0)
        analyzable = base & valid_price & valid_cap & valid_volume & valid_shares

        market_cap_usd = market_cap_source * 1_000.0
        daily_dollar_volume = price * volume
        turnover = volume / (shares_source * 1_000.0)
        implied_cap_ratio = market_cap_source / (price * shares_source)
        ratio_valid = base & valid_price & valid_cap & valid_shares

        analysis = pd.DataFrame(
            {
                "date": pd.to_datetime(frame["date"], errors="coerce"),
                "base": base,
                "analyzable": analyzable,
                "price": price,
                "market_cap_usd": market_cap_usd,
                "daily_dollar_volume": daily_dollar_volume,
                "turnover": turnover,
            }
        )
        eligible = analysis.loc[analyzable].copy()
        mcap_rank = eligible.groupby("date", sort=False)["market_cap_usd"].rank(
            method="average", pct=True
        )
        dollar_volume_rank = eligible.groupby(
            "date", sort=False
        )["daily_dollar_volume"].rank(method="average", pct=True)

        record = {
            "year": year,
            "development_period": year <= 2012,
            "all_security_day_rows": int(len(frame)),
            "base_common_share_rows": int(base.sum()),
            "base_unique_permnos": int(frame.loc[base, "permno"].nunique()),
            "us_incorporated_base_rows": int(us_incorporated.sum()),
            "analyzable_rows": int(analyzable.sum()),
            "missing_or_nonpositive_price_rows": int((base & ~valid_price).sum()),
            "missing_or_nonpositive_market_cap_rows": int((base & ~valid_cap).sum()),
            "missing_volume_rows": int((base & volume.isna()).sum()),
            "delisting_rows": int(frame["daily_delisting_flag"].eq("Y").sum()),
            "delisting_missing_return_rows": int(
                (
                    frame["daily_delisting_flag"].eq("Y")
                    & frame["return_total"].isna()
                ).sum()
            ),
            "price_ge_3_rows": int((analyzable & price.ge(3)).sum()),
            "price_ge_5_rows": int((analyzable & price.ge(5)).sum()),
            "mcap_ge_100m_rows": int(
                (analyzable & market_cap_usd.ge(100_000_000)).sum()
            ),
            "mcap_ge_300m_rows": int(
                (analyzable & market_cap_usd.ge(300_000_000)).sum()
            ),
            "daily_dollar_volume_ge_1m_rows": int(
                (analyzable & daily_dollar_volume.ge(1_000_000)).sum()
            ),
            "daily_dollar_volume_ge_5m_rows": int(
                (analyzable & daily_dollar_volume.ge(5_000_000)).sum()
            ),
            "combined_price5_mcap300m_daily_dv5m_rows": int(
                (
                    analyzable
                    & price.ge(5)
                    & market_cap_usd.ge(300_000_000)
                    & daily_dollar_volume.ge(5_000_000)
                ).sum()
            ),
            "above_mcap_date_pct10_rows": int(mcap_rank.gt(0.10).sum()),
            "above_mcap_date_pct20_rows": int(mcap_rank.gt(0.20).sum()),
            "above_daily_dv_date_pct10_rows": int(
                dollar_volume_rank.gt(0.10).sum()
            ),
            "above_daily_dv_date_pct20_rows": int(
                dollar_volume_rank.gt(0.20).sum()
            ),
        }
        annual_records.append(record)
        for name, values in [
            ("price_usd_candidate", eligible["price"]),
            ("market_cap_usd_candidate", eligible["market_cap_usd"]),
            ("daily_dollar_volume_usd_candidate", eligible["daily_dollar_volume"]),
            ("turnover_candidate", eligible["turnover"]),
        ]:
            quantile_records.extend(
                _quantile_records(
                    year,
                    name,
                    values,
                    development_period=year <= 2012,
                )
            )

        ratio = implied_cap_ratio.loc[ratio_valid].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        ratio_quantiles = ratio.quantile([0.01, 0.50, 0.99])
        unit_records.append(
            {
                "year": year,
                "rows": int(len(ratio)),
                "dlycap_div_price_times_shrout_p01": float(
                    ratio_quantiles.loc[0.01]
                ),
                "dlycap_div_price_times_shrout_median": float(
                    ratio_quantiles.loc[0.50]
                ),
                "dlycap_div_price_times_shrout_p99": float(
                    ratio_quantiles.loc[0.99]
                ),
                "share_and_cap_scale_consistent_with_thousands": bool(
                    0.95 <= ratio_quantiles.loc[0.50] <= 1.05
                ),
            }
        )

    annual = pd.DataFrame(annual_records)
    quantiles = pd.DataFrame(quantile_records)
    units = pd.DataFrame(unit_records)
    annual.to_csv(outputs["annual"], index=False)
    quantiles.to_csv(outputs["quantiles"], index=False)
    units.to_csv(outputs["units"], index=False)

    development = annual.loc[annual["development_period"]]
    monitor = annual.loc[~annual["development_period"]]
    unit_pass = bool(
        units["share_and_cap_scale_consistent_with_thousands"].all()
    )
    memo = f"""# Phase 2A data-foundation and universe profile

Generated: {datetime.now().isoformat(timespec="seconds")}

## Scope

This report is descriptive. It does not select a final universe, create labels,
inspect signal performance, or use 2013–2025 outcomes to tune a cutoff.

## Canonical data

- Canonical root: `{canonical_root}`
- Year partitions: {len(year_dirs)}
- Security-day rows: {int(annual["all_security_day_rows"].sum()):,}
- Base common-share rows: {int(annual["base_common_share_rows"].sum()):,}
- Base common-share PERMNO-year counts are reported in the annual table.
- Delisting rows retained: {int(annual["delisting_rows"].sum()):,}
- Delisting rows with missing total return: {int(annual["delisting_missing_return_rows"].sum()):,}

## Source-unit diagnostic

`DlyCap / (abs(DlyPrc) * ShrOut)` has an annual median near one in every
year: **{unit_pass}**. This is consistent with both `DlyCap` and the
price-times-`ShrOut` product being expressed in thousands of dollars. The
profile therefore reports `market_cap_source * 1,000` as a candidate USD
conversion. Vendor documentation remains the authoritative contract.

## Development-period candidate coverage (2000–2012)

- Analyzable rows: {int(development["analyzable_rows"].sum()):,}
- Price ≥ $5: {_weighted_rate(development, "price_ge_5_rows", "analyzable_rows"):.2%}
- Market cap ≥ $300M: {_weighted_rate(development, "mcap_ge_300m_rows", "analyzable_rows"):.2%}
- Daily dollar volume ≥ $5M: {_weighted_rate(development, "daily_dollar_volume_ge_5m_rows", "analyzable_rows"):.2%}
- Combined $5 / $300M / $5M-daily-volume diagnostic:
  {_weighted_rate(development, "combined_price5_mcap300m_daily_dv5m_rows", "analyzable_rows"):.2%}

Daily dollar volume is **not** ADV20. These figures only describe candidate
economic scales. Phase 2B must compute lagged ADV20 before any eligibility rule.

## Strict-OOS-period monitoring rows (2013–2025)

The annual table includes {int(monitor["analyzable_rows"].sum()):,} analyzable
rows for distribution-shift monitoring only. They are not used here to choose
the primary cutoff.

## Decision gate

No universe choice is frozen in Phase 2A. Phase 2B may compare the predeclared
absolute and date-percentile candidates using only the development period,
then lock one rule before any signal evaluation.
"""
    outputs["memo"].write_text(memo, encoding="utf-8")
    logger.info("Universe profile complete: %s", outputs)
    return outputs

