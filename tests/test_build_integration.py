from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from state_alpha.data.build import build_canonical
from state_alpha.data.ciz import SOURCE_ROW_COLUMN
from state_alpha.data.profile import profile_universe
from state_alpha.data.schema import SOURCE_COLUMNS
from state_alpha.data.validate import validate_canonical


def source_row(**updates: Any) -> dict[str, Any]:
    row = {column: None for column in SOURCE_COLUMNS}
    row.update(
        {
            "PERMNO": 10001,
            "PERMCO": 9001,
            "DlyCalDt": "2020-01-02",
            "PrimaryExch": "N",
            "ConditionalType": "RW",
            "TradingStatusFlg": "A",
            "SecurityType": "EQTY",
            "SecuritySubType": "COM",
            "ShareType": "NS",
            "USIncFlg": "Y",
            "IssuerType": "CORP",
            "DlyDelFlg": "N",
            "DlyPrc": 10.0,
            "DlyCap": 1000.0,
            "DlyRet": 0.01,
            "DlyRetx": 0.01,
            "DlyFacPrc": 1.0,
            "DlyVol": 1000.0,
            "DlyClose": 10.0,
            "DlyLow": 9.5,
            "DlyHigh": 10.2,
            "DlyOpen": 9.8,
            "ShrOut": 100.0,
            "vwretd": 0.002,
            "vwretx": 0.002,
            "ewretd": 0.001,
            "ewretx": 0.001,
            "sprtrn": 0.002,
        }
    )
    row.update(updates)
    return row


def make_project(tmp_path: Path) -> tuple[Path, Path]:
    project_1 = tmp_path / "project_1"
    project = project_1 / "project_2"
    raw = tmp_path / "raw" / "us" / "crsp" / "raw"
    raw.mkdir(parents=True)
    (project / "configs").mkdir(parents=True)
    for relative in [
        "data/interim",
        "data/processed",
        "artifacts/models",
        "reports/figures",
        "reports/tables",
        "reports/memo",
        "logs",
    ]:
        (project / relative).mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        "[project]\nname='test-project'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    paths = {
        "external": {
            "data_root": str(tmp_path / "raw"),
            "crsp_raw_dir": str(raw),
            "project_1_root": str(project_1),
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
    contract = {
        "version": 1,
        "source_glob": "*.zip",
        "chunk_size": 2,
        "outputs": {
            "root_name": "canonical_v1",
            "parquet_compression": "zstd",
            "parquet_compression_level": 1,
        },
    }
    (project / "configs" / "paths.yaml").write_text(
        yaml.safe_dump(paths),
        encoding="utf-8",
    )
    (project / "configs" / "data_contract.yaml").write_text(
        yaml.safe_dump(contract),
        encoding="utf-8",
    )
    return project, raw


def test_atomic_build_reconciles_and_refuses_overwrite(tmp_path: Path) -> None:
    project, raw = make_project(tmp_path)
    rows = [
        source_row(DisExDt="2020-01-02", DisSeqNbr=1, DisType="CD"),
        source_row(DisExDt="2020-01-02", DisSeqNbr=2, DisType="SD"),
        source_row(
            PERMNO=10002,
            PERMCO=9002,
            DlyCalDt="2020-01-03",
            DlyDelFlg="Y",
            TradingStatusFlg="D",
            SecuritySubType="UNK",
            DlyRet=None,
        ),
    ]
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    zip_path = raw / "sample.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="sample.csv")

    final_root = build_canonical(project)
    security = pd.read_parquet(final_root / "security_day")
    distributions = pd.read_parquet(final_root / "distributions")
    manifest = json.loads((final_root / "manifest.json").read_text(encoding="utf-8"))

    assert len(security) == 2
    assert len(distributions) == 2
    assert manifest["audit"]["raw_rows"] == 3
    assert manifest["audit"]["duplicate_rows_beyond_first"] == 1
    assert manifest["audit"]["delisting_rows"] == 1
    assert manifest["raw_sources_unchanged"] is True
    validation_path, validation = validate_canonical(project)
    assert validation_path.is_file()
    assert validation["passed"] is True
    profile_outputs = profile_universe(project)
    assert all(path.is_file() for path in profile_outputs.values())
    with pytest.raises(FileExistsError):
        build_canonical(project)
    with pytest.raises(FileExistsError):
        profile_universe(project)
