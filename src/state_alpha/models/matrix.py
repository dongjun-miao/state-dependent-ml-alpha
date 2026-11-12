"""Exact Gate 3 feature / Phase 2B label joins and family composites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_alpha.features.core import (
    CATEGORICAL_FEATURE_COLUMNS,
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    OUTPUT_IDENTIFIER_COLUMNS,
)


LABEL_COLUMNS = [
    "execution_valid",
    "label_end_session_id_10d",
    "fwd_total_return_10d",
    "fwd_market_excess_return_10d",
    "fwd_total_return_rank_10d",
]
FAMILY_COLUMNS = [
    "family_relative_strength",
    "family_price_trend",
    "family_oversold_recovery",
]
FAMILY_RANK_COLUMNS = [f"rank__{column}" for column in FAMILY_COLUMNS]


def alpha_contract(
    model_contract: dict[str, Any],
) -> tuple[list[str], dict[str, int], dict[str, list[str]]]:
    """Return ordered alpha names, orientation, and family membership."""

    names: list[str] = []
    directions: dict[str, int] = {}
    families: dict[str, list[str]] = {}
    for family, value in model_contract["alpha_families"].items():
        members = value["members"]
        family_names = list(members)
        families[family] = family_names
        for name, direction in members.items():
            if name in directions:
                raise ValueError(f"Alpha appears in multiple families: {name}")
            if int(direction) not in (-1, 1):
                raise ValueError(f"Alpha direction must be -1 or 1: {name}")
            names.append(name)
            directions[name] = int(direction)
    if any(name not in FEATURE_COLUMNS for name in names):
        raise ValueError("Model alpha contract references unknown Gate 3 feature")
    return names, directions, families


def _read_year(root: Path, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(root / f"year={year}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def load_modeling_year(
    feature_panel_root: Path,
    phase2b_panel_root: Path,
    year: int,
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    """Join one year exactly, filter valid primary labels, and rank alphas."""

    features = _read_year(feature_panel_root, year)
    labels = pd.read_parquet(
        phase2b_panel_root / f"year={year}",
        columns=[*OUTPUT_IDENTIFIER_COLUMNS, *LABEL_COLUMNS],
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="raise")
    try:
        pd.testing.assert_frame_equal(
            features[OUTPUT_IDENTIFIER_COLUMNS].reset_index(drop=True),
            labels[OUTPUT_IDENTIFIER_COLUMNS].reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(
            f"Gate 3 / Phase 2B identifier mismatch in {year}"
        ) from exc
    frame = features.merge(
        labels[
            [
                "permno",
                "date",
                "session_id",
                *LABEL_COLUMNS,
            ]
        ],
        on=["permno", "date", "session_id"],
        how="left",
        validate="one_to_one",
    )
    valid = (
        frame["universe_primary"].astype(bool)
        & frame["execution_valid"].astype(bool)
        & frame["fwd_total_return_rank_10d"].notna()
        & frame["fwd_total_return_10d"].notna()
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        return frame

    alpha_names, directions, families = alpha_contract(model_contract)
    grouped = frame.groupby("date", sort=False, observed=True)
    for name in alpha_names:
        frame[f"rank__{name}"] = grouped[name].rank(
            method="average",
            pct=True,
            ascending=directions[name] > 0,
        )

    family_name_map = {
        "relative_strength": "family_relative_strength",
        "price_trend": "family_price_trend",
        "oversold_recovery": "family_oversold_recovery",
    }
    for family, members in families.items():
        output_name = family_name_map[family]
        rank_names = [f"rank__{name}" for name in members]
        minimum = int(
            model_contract["alpha_families"][family][
                "minimum_nonmissing_members"
            ]
        )
        counts = frame[rank_names].notna().sum(axis=1)
        score = frame[rank_names].sub(0.5).mean(axis=1, skipna=True)
        frame[output_name] = score.where(counts.ge(minimum))
        frame[f"rank__{output_name}"] = frame.groupby(
            "date", sort=False, observed=True
        )[output_name].rank(method="average", pct=True)
    frame["base_family_score"] = frame[FAMILY_COLUMNS].mean(
        axis=1, skipna=False
    )
    return frame.sort_values(
        ["date", "permno"], kind="stable"
    ).reset_index(drop=True)


def modeling_sample(
    frame: pd.DataFrame,
    maximum_rows: int,
) -> pd.DataFrame:
    """Deterministic chronological-stride optimizer sample."""

    if maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    if len(frame) <= maximum_rows:
        sampled = frame.copy()
    else:
        stride = int(np.ceil(len(frame) / maximum_rows))
        sampled = frame.iloc[::stride].head(maximum_rows).copy()
    keep = [
        "permno",
        "date",
        "session_id",
        "label_end_session_id_10d",
        "fwd_total_return_10d",
        "fwd_total_return_rank_10d",
        *CONTINUOUS_FEATURE_COLUMNS,
        *CATEGORICAL_FEATURE_COLUMNS,
        *FAMILY_COLUMNS,
        "base_family_score",
    ]
    return sampled[keep].reset_index(drop=True)


def classical_sample(
    frame: pd.DataFrame,
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    """Apply the frozen complete-case and minimum breadth rules."""

    alpha_names, _, _ = alpha_contract(model_contract)
    classical = model_contract["classical_baselines"]
    valid = pd.Series(True, index=frame.index)
    if bool(classical["complete_case_across_alpha_signals"]):
        valid &= frame[alpha_names].notna().all(axis=1)
    output = frame.loc[valid].copy()
    minimum = int(classical["minimum_names_per_date"])
    date_counts = output.groupby(
        "date", sort=False, observed=True
    )["permno"].transform("size")
    output = output.loc[date_counts.ge(minimum)].copy()
    return output.reset_index(drop=True)


def period_name(year: int) -> str:
    if year <= 2009:
        return "initial_train"
    if year <= 2012:
        return "model_selection_validation"
    return "strict_oos"
