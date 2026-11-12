from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from state_alpha.config import load_yaml
from state_alpha.features.core import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    OUTPUT_IDENTIFIER_COLUMNS,
)
from state_alpha.models.gate4 import build_gate4
from state_alpha.models.matrix import alpha_contract
from state_alpha.models.validate_gate4 import validate_gate4


def _synthetic_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "synthetic_project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    for name in [
        "data_contract.yaml",
        "features.yaml",
        "labels.yaml",
        "models.yaml",
    ]:
        shutil.copy2(source_root / "configs" / name, configs / name)
    external = (tmp_path / "external").as_posix()
    configs.joinpath("paths.yaml").write_text(
        f"""version: 1
external:
  data_root: "{external}"
  project_1_root: "{external}/project_1"
project:
  root: "."
  data_interim: "data/interim"
  data_processed: "data/processed"
  model_artifacts: "artifacts/models"
  reports_figures: "reports/figures"
  reports_tables: "reports/tables"
  reports_memo: "reports/memo"
  logs: "logs"
path_policy:
  allowed_runtime_write_roots:
    - "data/interim"
    - "data/processed"
    - "artifacts/models"
    - "reports/figures"
    - "reports/tables"
    - "reports/memo"
    - "logs"
""",
        encoding="utf-8",
    )

    features_root = project / "data" / "interim" / "features_v1"
    feature_panel = features_root / "panel"
    labels_root = project / "data" / "interim" / "phase2b_v1"
    label_panel = labels_root / "panel"
    feature_panel.mkdir(parents=True)
    label_panel.mkdir(parents=True)

    market_dates = pd.bdate_range("2000-01-03", "2025-12-31")
    calendar = pd.DataFrame(
        {
            "date": market_dates,
            "session_id": np.arange(1, len(market_dates) + 1, dtype=np.int32),
        }
    )
    calendar.to_parquet(labels_root / "market_calendar.parquet", index=False)
    market = calendar.copy()
    angle = np.arange(len(market), dtype=float) / 80.0
    market["market_return_60d"] = np.sin(angle) * 0.10
    market["market_realized_volatility_20d"] = (
        0.15 + np.cos(angle) * 0.03
    )
    market.to_parquet(
        features_root / "market_features.parquet", index=False
    )

    contract = load_yaml(configs / "models.yaml")
    _, directions, _ = alpha_contract(contract)
    rng = np.random.default_rng(20260731)
    names = 210
    permnos = np.arange(10_000, 10_000 + names, dtype=np.int64)
    base_by_name = np.linspace(-1.0, 1.0, names)
    for year in range(2001, 2026):
        year_calendar = calendar.loc[calendar["date"].dt.year.eq(year)].head(3)
        index = pd.MultiIndex.from_product(
            [year_calendar["date"], permnos],
            names=["date", "permno"],
        ).to_frame(index=False)
        index["permco"] = index["permno"] + 50_000
        session_map = year_calendar.set_index("date")["session_id"]
        index["session_id"] = (
            index["date"].map(session_map).astype("int32")
        )
        index["universe_primary"] = True
        index["largest_lagged_mcap_share_class"] = True
        base = np.tile(base_by_name, len(year_calendar))
        time_component = np.repeat(
            np.linspace(-0.1, 0.1, len(year_calendar)), names
        )

        features = index[OUTPUT_IDENTIFIER_COLUMNS].copy()
        for number, column in enumerate(CONTINUOUS_FEATURE_COLUMNS):
            if column in directions:
                features[column] = (
                    directions[column] * base
                    + rng.normal(0.0, 0.08, len(index))
                )
            else:
                features[column] = (
                    base * (0.05 + (number % 7) * 0.02)
                    + time_component
                    + rng.normal(0.0, 0.12, len(index))
                )
        features["industry_code_2digit"] = (
            features["permno"] % 10
        ).astype("int16")
        features["exchange"] = np.select(
            [
                features["permno"].mod(3).eq(0),
                features["permno"].mod(3).eq(1),
            ],
            ["N", "Q"],
            default="A",
        )
        features = features[
            [*OUTPUT_IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
        ]

        labels = index[OUTPUT_IDENTIFIER_COLUMNS].copy()
        labels["execution_valid"] = True
        labels["label_end_session_id_10d"] = (
            labels["session_id"] + 11
        ).astype("int32")
        realized = (
            0.025 * base
            + 0.002 * time_component
            + rng.normal(0.0, 0.002, len(index))
        )
        labels["fwd_total_return_10d"] = realized
        labels["fwd_market_excess_return_10d"] = realized
        labels["fwd_total_return_rank_10d"] = (
            labels.groupby("date")["fwd_total_return_10d"].rank(pct=True)
            - 0.5
        )

        feature_partition = feature_panel / f"year={year}"
        label_partition = label_panel / f"year={year}"
        feature_partition.mkdir()
        label_partition.mkdir()
        features.to_parquet(
            feature_partition / "part-00000.parquet", index=False
        )
        labels.to_parquet(
            label_partition / "part-00000.parquet", index=False
        )
    return project


def test_gate4_atomic_build_and_independent_validation(
    tmp_path: Path,
) -> None:
    project = _synthetic_project(tmp_path)
    output = build_gate4(project)
    assert output == project / "data" / "processed" / "gate4_v1"
    assert output.is_dir()
    assert not list(
        (project / "data" / "processed").glob("_gate4_v1_build_*")
    )

    result = validate_gate4(project)
    assert result["passed"], result["errors"]
    assert result["model_refits_verified"] == 14
    assert result["strict_oos_annual_refits_verified"] == 13
    assert result["model_states_verified"] == 14
    assert result["all_refits_embargoed"] is True
    assert result["validation_prediction_rows"] == 3 * 3 * 210
    assert result["strict_oos_prediction_rows"] == 13 * 3 * 210
    assert result["strict_oos_used_for_selection"] is False
    assert result["tree_model_fit"] is False
