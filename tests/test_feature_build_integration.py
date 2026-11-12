from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from state_alpha.features.build import build_features
from state_alpha.features.core import OUTPUT_IDENTIFIER_COLUMNS
from state_alpha.features.validate import validate_features


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    for relative in [
        "configs",
        "data/interim/canonical_v1/security_day",
        "data/interim/phase2b_v1/panel",
        "data/processed",
        "artifacts/models",
        "reports/figures",
        "reports/tables",
        "reports/memo",
        "logs",
    ]:
        (project / relative).mkdir(parents=True, exist_ok=True)
    (project / "pyproject.toml").write_text(
        "[project]\nname='feature-test'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    paths = {
        "external": {
            "data_root": str(tmp_path / "external"),
            "project_1_root": str(tmp_path / "project_1"),
        },
        "project": {
            "root": ".",
            "data_interim": "data/interim",
            "data_processed": "data/processed",
            "model_artifacts": "artifacts/models",
            "reports_figures": "reports/figures",
            "reports_tables": "reports/tables",
            "reports_memo": "reports/memo",
            "logs": "logs",
        },
        "path_policy": {
            "allowed_runtime_write_roots": [
                "data/interim",
                "data/processed",
                "artifacts/models",
                "reports/figures",
                "reports/tables",
                "reports/memo",
                "logs",
            ]
        },
    }
    (project / "configs" / "paths.yaml").write_text(
        yaml.safe_dump(paths), encoding="utf-8"
    )
    (project / "configs" / "data_contract.yaml").write_text(
        "version: 1\n", encoding="utf-8"
    )
    source_config = (
        Path(__file__).resolve().parents[1] / "configs" / "features.yaml"
    )
    shutil.copy2(source_config, project / "configs" / "features.yaml")
    return project


def _write_inputs(project: Path) -> None:
    dates = pd.bdate_range("2000-01-03", periods=330)
    calendar = pd.DataFrame(
        {
            "session_id": np.arange(len(dates), dtype=np.int32),
            "date": dates,
            "market_vw_return": np.sin(np.arange(len(dates)) / 23.0) * 0.001,
        }
    )
    calendar.to_parquet(
        project / "data/interim/phase2b_v1/market_calendar.parquet",
        index=False,
    )
    security_rows: list[pd.DataFrame] = []
    for permno in [10001, 10002]:
        returns = (
            0.0005
            + (permno - 10001) * 0.0001
            + np.sin(np.arange(len(dates)) / 19.0) * 0.002
        )
        security_rows.append(
            pd.DataFrame(
                {
                    "permno": permno,
                    "date": dates,
                    "return_total": returns,
                    "return_ex_distribution": returns - 0.00005,
                    "price_raw": 25.0,
                    "volume": 2_000_000.0,
                    "shares_outstanding_source": 100_000.0,
                    "bid_raw": 24.99,
                    "ask_raw": 25.01,
                    "siccd": 3571,
                    "primary_exchange": "N",
                }
            )
        )
    security = pd.concat(security_rows, ignore_index=True)
    security_root = project / "data/interim/canonical_v1/security_day"
    for year, group in security.groupby(security["date"].dt.year):
        partition = security_root / f"year={year}"
        partition.mkdir()
        group.to_parquet(partition / "part-00000.parquet", index=False)

    panel_root = project / "data/interim/phase2b_v1/panel"
    empty = pd.DataFrame(
        {
            "permno": pd.Series(dtype="int64"),
            "permco": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "session_id": pd.Series(dtype="int32"),
            "universe_primary": pd.Series(dtype="bool"),
            "largest_lagged_mcap_share_class": pd.Series(dtype="bool"),
            "lag_market_cap_usd": pd.Series(dtype="float64"),
            "lag_adv20_usd": pd.Series(dtype="float64"),
        }
    )
    for year in range(2000, 2026):
        partition = panel_root / f"year={year}"
        partition.mkdir()
        frame = empty.copy()
        if year == 2001:
            selected_dates = dates[dates.year == 2001][-40:]
            frame = pd.MultiIndex.from_product(
                [[10001, 10002], selected_dates],
                names=["permno", "date"],
            ).to_frame(index=False)
            frame["permco"] = frame["permno"] + 1_000
            frame["session_id"] = frame["date"].map(
                calendar.set_index("date")["session_id"]
            ).astype("int32")
            frame["universe_primary"] = True
            frame["largest_lagged_mcap_share_class"] = True
            frame["lag_market_cap_usd"] = 500_000_000.0
            frame["lag_adv20_usd"] = 5_000_000.0
            frame = frame.sort_values(
                ["date", "permno"], kind="stable"
            ).reset_index(drop=True)
        frame.to_parquet(partition / "part-00000.parquet", index=False)


def test_atomic_feature_build_and_independent_validation(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    _write_inputs(project)
    feature_root = build_features(project)
    result = validate_features(project)
    output = pd.read_parquet(feature_root / "panel/year=2001")
    input_keys = pd.read_parquet(
        project / "data/interim/phase2b_v1/panel/year=2001",
        columns=OUTPUT_IDENTIFIER_COLUMNS,
    )
    assert result["passed"] is True
    assert result["feature_count"] == 38
    assert result["label_columns"] == 0
    assert len(output) == len(input_keys) == 80
    assert not any(column.startswith("fwd_") for column in output)
