from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from state_alpha.config import load_yaml
from state_alpha.features.core import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    OUTPUT_IDENTIFIER_COLUMNS,
)
from state_alpha.models.linear import Gate4Preprocessor
from state_alpha.models.matrix import (
    FAMILY_COLUMNS,
    alpha_contract,
    classical_sample,
    load_modeling_year,
    modeling_sample,
)
from state_alpha.models.statistics import (
    daily_double_sorts,
    daily_fama_macbeth,
    daily_rank_ic,
    daily_sorts,
    newey_west_mean,
)


def _contract() -> dict:
    root = Path(__file__).resolve().parents[1]
    return load_yaml(root / "configs" / "models.yaml")


def _write_modeling_inputs(
    tmp_path: Path,
    *,
    year: int = 2005,
    dates: int = 8,
    names: int = 300,
) -> tuple[Path, Path]:
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    (feature_root / f"year={year}").mkdir(parents=True)
    (label_root / f"year={year}").mkdir(parents=True)
    date_values = pd.bdate_range(f"{year}-01-03", periods=dates)
    index = pd.MultiIndex.from_product(
        [date_values, np.arange(10_000, 10_000 + names)],
        names=["date", "permno"],
    ).to_frame(index=False)
    index["permco"] = index["permno"] + 50_000
    session_map = {
        date: number for number, date in enumerate(date_values, start=1_000)
    }
    index["session_id"] = index["date"].map(session_map).astype("int32")
    index["universe_primary"] = True
    index["largest_lagged_mcap_share_class"] = True
    rng = np.random.default_rng(20260731)
    base = np.tile(np.linspace(-1.0, 1.0, names), dates)
    features = index[OUTPUT_IDENTIFIER_COLUMNS].copy()
    for number, column in enumerate(CONTINUOUS_FEATURE_COLUMNS):
        features[column] = (
            base * (0.2 + number / 100.0)
            + rng.normal(0.0, 0.2, len(features))
        )
    features["industry_code_2digit"] = (
        features["permno"] % 90
    ).astype("int16")
    features["exchange"] = np.where(
        features["permno"].mod(3).eq(0),
        "N",
        np.where(features["permno"].mod(3).eq(1), "Q", "A"),
    )
    features = features[[*OUTPUT_IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]]
    labels = index[OUTPUT_IDENTIFIER_COLUMNS].copy()
    labels["execution_valid"] = True
    labels["label_end_session_id_10d"] = labels["session_id"] + 11
    labels["fwd_total_return_10d"] = base * 0.03
    labels["fwd_market_excess_return_10d"] = base * 0.03
    labels["fwd_total_return_rank_10d"] = (
        labels.groupby("date")["fwd_total_return_10d"].rank(pct=True) - 0.5
    )
    features.to_parquet(
        feature_root / f"year={year}" / "part-00000.parquet",
        index=False,
    )
    labels.to_parquet(
        label_root / f"year={year}" / "part-00000.parquet",
        index=False,
    )
    return feature_root, label_root


def test_exact_join_and_family_composites(tmp_path: Path) -> None:
    feature_root, label_root = _write_modeling_inputs(tmp_path)
    frame = load_modeling_year(
        feature_root, label_root, 2005, _contract()
    )
    assert len(frame) == 2_400
    assert frame[FAMILY_COLUMNS].notna().all().all()
    assert frame["base_family_score"].between(-0.5, 0.5).all()
    frame.loc[0, "rs_excess_return_20d"] = np.nan
    classical = classical_sample(frame, _contract())
    assert len(classical) == len(frame) - 1
    sample = modeling_sample(frame, 500)
    assert len(sample) <= 500
    assert sample["date"].is_monotonic_increasing


def test_identifier_mismatch_fails_closed(tmp_path: Path) -> None:
    feature_root, label_root = _write_modeling_inputs(tmp_path)
    path = label_root / "year=2005" / "part-00000.parquet"
    labels = pd.read_parquet(path)
    labels.loc[0, "permno"] += 1
    labels.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="identifier mismatch"):
        load_modeling_year(feature_root, label_root, 2005, _contract())


def test_classical_statistics_are_ordered_and_interpretable(
    tmp_path: Path,
) -> None:
    feature_root, label_root = _write_modeling_inputs(tmp_path)
    contract = _contract()
    frame = load_modeling_year(
        feature_root, label_root, 2005, contract
    )
    alpha_names, directions, _ = alpha_contract(contract)
    rank_map = {name: f"rank__{name}" for name in alpha_names}
    rank_map.update(
        {family: f"rank__{family}" for family in FAMILY_COLUMNS}
    )
    ic = daily_rank_ic(frame, rank_map)
    assert len(ic) == 8 * len(rank_map)
    family_ic = ic.loc[ic["signal"].isin(FAMILY_COLUMNS), "rank_ic"]
    assert family_ic.mean() > 0.45
    alpha_ic = ic.loc[ic["signal"].isin(alpha_names)].groupby(
        "signal"
    )["rank_ic"].mean()
    assert all(alpha_ic[name] * directions[name] > 0 for name in alpha_names)
    quantiles = {name: [10] for name in rank_map}
    sorts = daily_sorts(
        frame,
        rank_map,
        quantiles,
        turnover_signals=set(FAMILY_COLUMNS),
    )
    assert sorts["spread_return"].mean() > 0
    double = daily_double_sorts(
        frame, ["log_lagged_market_cap"]
    )
    assert not double.empty
    fmb = daily_fama_macbeth(frame, contract)
    assert {"base", "explicit_interactions"} == set(
        fmb["specification"]
    )
    assert np.isfinite(
        fmb.loc[fmb["term"].eq("family_relative_strength"), "coefficient"]
    ).all()


def test_preprocessor_uses_training_moments_only(tmp_path: Path) -> None:
    feature_root, label_root = _write_modeling_inputs(tmp_path)
    contract = _contract()
    frame = load_modeling_year(
        feature_root, label_root, 2005, contract
    )
    training = frame.iloc[:1_200].copy()
    validation = frame.iloc[1_200:].copy()
    preprocessor = Gate4Preprocessor.fit(training, contract)
    upper_before = preprocessor.upper.copy()
    validation.loc[:, CONTINUOUS_FEATURE_COLUMNS] = 1e12
    transformed = preprocessor.transform(validation)
    assert np.array_equal(preprocessor.upper, upper_before)
    assert transformed.shape[1] == len(preprocessor.feature_names) == 174
    assert np.isfinite(transformed.data).all()


def test_newey_west_mean_handles_serial_series() -> None:
    values = np.sin(np.arange(250) / 20.0) + 0.05
    result = newey_west_mean(values, lags=20)
    assert result["observations"] == 250
    assert np.isfinite(result["newey_west_t"])
