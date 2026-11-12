from __future__ import annotations

import numpy as np
import pandas as pd

from state_alpha.config import load_yaml
from state_alpha.portfolio.core import (
    exposure_matrix,
    performance_summary,
    raw_long_short_weights,
    realized_return_paths,
    turnover_between,
    variant_catalog,
    weight_matrix,
)
from state_alpha.portfolio.gate6 import _write_evidence_memo


def _contract() -> dict:
    from pathlib import Path

    return load_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "gate6.yaml"
    )


def test_variant_catalog_carries_all_scores_without_selection() -> None:
    catalog = variant_catalog(_contract())
    assert len(catalog) == 100
    assert catalog["signal"].nunique() == 10
    assert len(catalog.loc[catalog["bucket"].eq("all")]) == 40
    assert len(catalog.loc[catalog["bucket"].ne("all")]) == 60


def test_equal_and_rank_weights_are_one_gross_and_zero_net() -> None:
    scores = pd.Series(np.arange(100, dtype=float))
    eligible = np.ones(100, dtype=bool)
    for weighting in ["equal_weight", "rank_weight"]:
        weights, long_count, short_count = raw_long_short_weights(
            scores,
            weighting=weighting,
            eligible=eligible,
            zero_is_cash=False,
        )
        assert long_count == 10
        assert short_count == 10
        assert np.isclose(weights.sum(), 0.0)
        assert np.isclose(np.abs(weights).sum(), 1.0)


def test_neutralized_variants_hit_prefill_constraints() -> None:
    rng = np.random.default_rng(7)
    count = 500
    frame = pd.DataFrame(
        {
            "beta_252d": rng.normal(size=count),
            "log_lagged_market_cap": rng.normal(size=count),
            "log_price": rng.normal(size=count),
            "log_adv20_usd": rng.normal(size=count),
            "amihud_illiquidity_20d": rng.normal(size=count),
            "quoted_spread_proxy": rng.normal(size=count),
            "industry_code_2digit": np.arange(count) % 100,
        }
    )
    for number, signal in enumerate(_contract()["scores"]):
        frame[signal] = rng.normal(size=count) + number * 1e-5
    catalog = variant_catalog(_contract()).loc[
        lambda value: value["bucket"].eq("all")
    ].reset_index(drop=True)
    weights, diagnostics, _, _ = weight_matrix(
        frame, catalog, _contract()
    )
    _, _, constraints = exposure_matrix(frame)
    neutral = catalog["neutralization"].eq(
        "risk_neutralized"
    ).to_numpy()
    assert np.allclose(
        np.abs(weights[neutral]).sum(axis=1), 1.0
    )
    assert np.max(np.abs(weights[neutral] @ constraints)) < 1e-8
    assert (
        diagnostics.loc[
            neutral, "prefill_max_abs_constraint"
        ].max()
        < 1e-8
    )


def test_realized_paths_handle_fill_terminal_and_missing_without_filtering() -> None:
    permnos = np.asarray([1, 2, 3])
    returns = {
        12: pd.DataFrame(
            {
                "return_total": [0.10, 0.20, np.nan],
                "daily_delisting_flag": ["N", "Y", "Y"],
            },
            index=permnos,
        ),
        13: pd.DataFrame(
            {
                "return_total": [0.05],
                "daily_delisting_flag": ["N"],
            },
            index=[1],
        ),
    }
    lower, upper, alive, active = realized_return_paths(
        permnos,
        np.asarray([True, True, True]),
        [12, 13],
        returns,
    )
    assert np.allclose(lower[0], [0.10, 0.05])
    assert np.allclose(lower[1], [0.20, 0.0])
    assert np.allclose(lower[2], [-1.0, 0.0])
    assert np.allclose(upper[2], [0.0, 0.0])
    assert active[1].tolist() == [True, False]
    assert not alive[1] and not alive[2]
    short_missing = np.asarray([0.0, 0.0, -0.5])
    assert float(short_missing @ upper[:, 0]) < float(
        short_missing @ lower[:, 0]
    )


def test_turnover_nets_simultaneous_entry_and_exit_by_security() -> None:
    entry = np.asarray([[0.25, -0.25]])
    exiting = np.asarray([[0.10, -0.20]])
    turnover = turnover_between(
        np.asarray([1, 2]),
        entry,
        np.asarray([1, 3]),
        exiting,
    )
    assert np.allclose(turnover, [0.5 * (0.15 + 0.25 + 0.20)])


def test_performance_summary_does_not_annualize_overlapping_labels() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    daily = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "variant_id": "x",
            "return": 0.001,
        }
    )
    summary = performance_summary(
        daily,
        value_column="return",
        group_columns=["variant_id"],
        periods={"sample": (2020, 2020)},
        hac_lags=5,
        trading_days=252,
    )
    assert np.isclose(summary.loc[0, "annualized_return"], 0.252)


def test_gate6_evidence_memo_is_materialized(tmp_path) -> None:
    performance = pd.DataFrame(
        {
            "period": ["boundary_clean_primary"] * 2,
            "weighting": ["rank_weight"] * 2,
            "neutralization": ["raw", "risk_neutralized"],
            "bucket": ["all"] * 2,
            "signal": ["base_family_score"] * 2,
            "annualized_return": [0.01, 0.01],
            "annualized_volatility": [0.02, 0.02],
            "sharpe": [0.5, 0.5],
            "newey_west_t": [1.0, 1.0],
            "maximum_drawdown": [-0.03, -0.03],
            "break_even_one_way_bps": [5.0, 5.0],
        }
    )
    exposure = pd.DataFrame(
        {
            "neutralization": ["risk_neutralized"],
            "bucket": ["all"],
            "maximum_prefill_constraint": [1e-12],
        }
    )
    audit = pd.DataFrame(
        {
            "point_in_time_primary_rows": [100],
            "formation_dates": [10],
            "label_filter_excluded_rows": [2],
        }
    )
    sensitivity = pd.DataFrame(
        {
            "period": ["boundary_clean_primary"],
            "annualized_return_difference_zero_minus_loss100": [0.001],
        }
    )
    path = tmp_path / "memo.md"
    _write_evidence_memo(
        path,
        performance=performance,
        exposure=exposure,
        formation_audit=audit,
        missing_sensitivity=sensitivity,
        config_hash="abc",
    )
    assert path.is_file()
    assert "Gate 6 portfolio evidence" in path.read_text(encoding="utf-8")
