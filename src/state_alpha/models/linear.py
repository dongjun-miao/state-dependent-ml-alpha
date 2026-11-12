"""Training-only preprocessing and explicit linear interaction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge

from state_alpha.features.core import CONTINUOUS_FEATURE_COLUMNS
from state_alpha.models.matrix import FAMILY_COLUMNS


CONTINUOUS_MODEL_COLUMNS = [
    *CONTINUOUS_FEATURE_COLUMNS,
    *FAMILY_COLUMNS,
]


@dataclass
class Gate4Preprocessor:
    """Frozen numeric moments and fixed categorical encoding."""

    continuous_columns: list[str]
    family_columns: list[str]
    interaction_state_columns: list[str]
    lower: np.ndarray
    upper: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        contract: dict[str, Any],
    ) -> "Gate4Preprocessor":
        linear = contract["linear_models"]
        continuous = list(CONTINUOUS_MODEL_COLUMNS)
        interaction_states = list(
            linear["explicit_interactions"]["state_features"]
        )
        missing = sorted(
            set([*continuous, *interaction_states]).difference(frame.columns)
        )
        if missing:
            raise ValueError(
                f"Training sample is missing linear-model columns: {missing}"
            )
        values = frame[continuous].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        percentiles = linear["preprocessing"]["winsorization_percentiles"]
        lower = np.nanpercentile(values, float(percentiles[0]), axis=0)
        upper = np.nanpercentile(values, float(percentiles[1]), axis=0)
        median = np.nanmedian(values, axis=0)
        if not (
            np.isfinite(lower).all()
            and np.isfinite(upper).all()
            and np.isfinite(median).all()
        ):
            invalid = [
                name
                for name, low, high, middle in zip(
                    continuous, lower, upper, median, strict=True
                )
                if not (
                    np.isfinite(low)
                    and np.isfinite(high)
                    and np.isfinite(middle)
                )
            ]
            raise ValueError(
                f"Training sample has unusable all-missing columns: {invalid}"
            )
        clipped = np.clip(values, lower, upper)
        filled = np.where(np.isfinite(clipped), clipped, median)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0, ddof=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(
            continuous_columns=continuous,
            family_columns=list(FAMILY_COLUMNS),
            interaction_state_columns=interaction_states,
            lower=lower,
            upper=upper,
            median=median,
            mean=mean,
            scale=scale,
        )

    @property
    def feature_names(self) -> list[str]:
        interactions = [
            f"{family}_x_{state}"
            for family in self.family_columns
            for state in self.interaction_state_columns
        ]
        industry = [f"industry_{value:02d}" for value in range(100)]
        industry.append("industry_missing")
        exchange = [
            "exchange_N",
            "exchange_Q",
            "exchange_A",
            "exchange_missing",
        ]
        return [
            *self.continuous_columns,
            *interactions,
            *industry,
            *exchange,
        ]

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        values = frame[self.continuous_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        values = np.clip(values, self.lower, self.upper)
        values = np.where(np.isfinite(values), values, self.median)
        standardized = ((values - self.mean) / self.scale).astype(
            np.float32
        )
        column_index = {
            name: index
            for index, name in enumerate(self.continuous_columns)
        }
        interaction_blocks = []
        for family in self.family_columns:
            family_values = standardized[:, column_index[family]][:, None]
            states = standardized[
                :,
                [
                    column_index[name]
                    for name in self.interaction_state_columns
                ],
            ]
            interaction_blocks.append(family_values * states)
        interactions = np.concatenate(interaction_blocks, axis=1).astype(
            np.float32
        )

        industry = pd.to_numeric(
            frame["industry_code_2digit"], errors="coerce"
        ).to_numpy(dtype=float)
        industry_code = np.full(len(frame), 100, dtype=np.int64)
        valid_industry = (
            np.isfinite(industry) & (industry >= 0) & (industry <= 99)
        )
        industry_code[valid_industry] = industry[valid_industry].astype(
            np.int64
        )
        exchange_map = {"N": 0, "Q": 1, "A": 2}
        exchange_code = (
            frame["exchange"].map(exchange_map).fillna(3).to_numpy(dtype=np.int64)
        )
        rows = np.arange(len(frame), dtype=np.int64)
        categorical = sparse.csr_matrix(
            (
                np.ones(len(frame) * 2, dtype=np.float32),
                (
                    np.concatenate([rows, rows]),
                    np.concatenate([industry_code, 101 + exchange_code]),
                ),
            ),
            shape=(len(frame), 105),
        )
        dense_block = np.concatenate(
            [standardized, interactions], axis=1
        )
        return sparse.hstack(
            [sparse.csr_matrix(dense_block), categorical],
            format="csr",
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "continuous_columns": self.continuous_columns,
            "family_columns": self.family_columns,
            "interaction_state_columns": self.interaction_state_columns,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "feature_names": self.feature_names,
        }


def _mean_daily_spearman(
    predictions: np.ndarray,
    labels: pd.Series,
    dates: pd.Series,
) -> tuple[float, int]:
    work = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="raise").to_numpy(),
            "prediction": np.asarray(predictions, dtype=float),
            "label": pd.to_numeric(labels, errors="coerce").to_numpy(),
        }
    ).dropna()
    work["prediction_rank"] = work.groupby(
        "date", sort=False, observed=True
    )["prediction"].rank(method="average", pct=True)
    grouped = work.groupby("date", sort=False, observed=True)
    count = grouped.size().astype(float)
    sx = grouped["prediction_rank"].sum()
    sy = grouped["label"].sum()
    sxx = work["prediction_rank"].pow(2).groupby(work["date"]).sum()
    syy = work["label"].pow(2).groupby(work["date"]).sum()
    sxy = (
        work["prediction_rank"].mul(work["label"])
        .groupby(work["date"])
        .sum()
    )
    numerator = sxy - sx * sy / count
    denominator = np.sqrt(
        (sxx - sx.pow(2) / count) * (syy - sy.pow(2) / count)
    )
    daily = numerator / denominator.where(denominator.gt(0))
    return float(daily.mean()), int(daily.notna().sum())


def _ridge(alpha: float, solver: str) -> Ridge:
    return Ridge(alpha=float(alpha), solver=solver, fit_intercept=True)


def _elastic_net(
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
) -> ElasticNet:
    return ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=int(max_iter),
        tol=float(tolerance),
        selection="cyclic",
    )


def select_hyperparameters(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Use only 2010-2012 validation rank IC to select linear penalties."""

    preprocessor = Gate4Preprocessor.fit(training, contract)
    x_train = preprocessor.transform(training)
    x_validation = preprocessor.transform(validation)
    y_train = pd.to_numeric(
        training["fwd_total_return_rank_10d"], errors="raise"
    ).to_numpy(dtype=float)
    linear = contract["linear_models"]
    records: list[dict[str, Any]] = []

    ridge_config = linear["ridge"]
    for alpha in ridge_config["alpha_grid"]:
        model = _ridge(alpha, ridge_config["solver"])
        model.fit(x_train, y_train)
        predictions = model.predict(x_validation)
        metric, dates = _mean_daily_spearman(
            predictions,
            validation["fwd_total_return_rank_10d"],
            validation["date"],
        )
        records.append(
            {
                "model": "ridge",
                "alpha": float(alpha),
                "l1_ratio": np.nan,
                "validation_mean_daily_rank_ic": metric,
                "validation_dates": dates,
                "iterations": np.nan,
            }
        )

    elastic_config = linear["elastic_net"]
    for alpha in elastic_config["alpha_grid"]:
        for l1_ratio in elastic_config["l1_ratio_grid"]:
            model = _elastic_net(
                alpha,
                l1_ratio,
                elastic_config["max_iter"],
                elastic_config["tolerance"],
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)
            predictions = model.predict(x_validation)
            metric, dates = _mean_daily_spearman(
                predictions,
                validation["fwd_total_return_rank_10d"],
                validation["date"],
            )
            records.append(
                {
                    "model": "elastic_net",
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "validation_mean_daily_rank_ic": metric,
                    "validation_dates": dates,
                    "iterations": int(model.n_iter_),
                }
            )
    results = pd.DataFrame(records)
    selected: dict[str, Any] = {}
    for model_name in ["ridge", "elastic_net"]:
        candidates = results.loc[results["model"].eq(model_name)].sort_values(
            [
                "validation_mean_daily_rank_ic",
                "alpha",
                "l1_ratio",
            ],
            ascending=[False, False, False],
            kind="stable",
        )
        if candidates.empty or candidates.iloc[0][
            "validation_mean_daily_rank_ic"
        ] != candidates.iloc[0]["validation_mean_daily_rank_ic"]:
            raise ValueError(f"No valid hyperparameter result for {model_name}")
        selected[model_name] = candidates.iloc[0].to_dict()
    return selected, results


def fit_selected_models(
    training: pd.DataFrame,
    selected: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[Gate4Preprocessor, dict[str, Ridge | ElasticNet]]:
    preprocessor = Gate4Preprocessor.fit(training, contract)
    x_train = preprocessor.transform(training)
    y_train = pd.to_numeric(
        training["fwd_total_return_rank_10d"], errors="raise"
    ).to_numpy(dtype=float)
    linear = contract["linear_models"]
    ridge = _ridge(
        selected["ridge"]["alpha"],
        linear["ridge"]["solver"],
    )
    ridge.fit(x_train, y_train)
    elastic = _elastic_net(
        selected["elastic_net"]["alpha"],
        selected["elastic_net"]["l1_ratio"],
        linear["elastic_net"]["max_iter"],
        linear["elastic_net"]["tolerance"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        elastic.fit(x_train, y_train)
    return preprocessor, {"ridge": ridge, "elastic_net": elastic}


def predict_models(
    frame: pd.DataFrame,
    preprocessor: Gate4Preprocessor,
    models: dict[str, Ridge | ElasticNet],
    *,
    chunk_size: int = 100_000,
) -> dict[str, np.ndarray]:
    predictions = {
        name: np.empty(len(frame), dtype=np.float32)
        for name in models
    }
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        chunk = frame.iloc[start:stop]
        x = preprocessor.transform(chunk)
        for name, model in models.items():
            predictions[name][start:stop] = model.predict(x).astype(
                np.float32
            )
    return predictions


def coefficient_table(
    models: dict[str, Ridge | ElasticNet],
    preprocessor: Gate4Preprocessor,
    *,
    evaluation_year: int,
    train_end_session_id: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    names = preprocessor.feature_names
    for model_name, model in models.items():
        coefficients = np.asarray(model.coef_, dtype=float)
        if len(coefficients) != len(names):
            raise ValueError("Coefficient and feature-name counts differ")
        for name, value in zip(names, coefficients, strict=True):
            records.append(
                {
                    "evaluation_year": evaluation_year,
                    "train_end_session_id": train_end_session_id,
                    "model": model_name,
                    "term": name,
                    "coefficient": value,
                }
            )
        records.append(
            {
                "evaluation_year": evaluation_year,
                "train_end_session_id": train_end_session_id,
                "model": model_name,
                "term": "intercept",
                "coefficient": float(model.intercept_),
            }
        )
    return pd.DataFrame(records)
