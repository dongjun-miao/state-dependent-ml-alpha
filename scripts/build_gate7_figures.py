"""Build the seven audited Gate 7 figures from frozen Gate 5/6 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#102A43"
BLUE = "#2F6B9A"
TEAL = "#0F7C7B"
ORANGE = "#E58B2A"
RED = "#C94C4C"
GOLD = "#C7A43A"
GRAY = "#7A8793"
LIGHT = "#EAF0F4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/gate7.yaml"))
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#B8C2CC",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#DCE3E8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def score_color(signal: str) -> str:
    if signal == "base_family_score":
        return GRAY
    if signal.startswith("ridge"):
        return TEAL
    if signal.startswith("tree_") and "gate" not in signal:
        return BLUE
    return ORANGE


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def figure_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    stages = [
        ("1", "CRSP CIZ", "Immutable raw data\nDelisting-aware returns"),
        ("2", "Point-in-time\npanel", "Lagged eligibility\nNo survivorship filter"),
        ("3", "Alpha x\nstate", "Price signals\nseparate from\nconditioning state"),
        ("4", "Purged\nwalk-forward", "Annual frozen fits\n10-session embargo"),
        ("5", "Formation\nrepair", "Rescore full t\nuniverse; rank\nbefore fill status"),
        ("6", "Delayed\nportfolios", "Close t signal\nClose t+1 execution"),
        ("7", "Implementation\naudit", "Cost and exposure;\nliquidity and ADV"),
    ]
    xs = np.linspace(0.2, 11.35, len(stages))
    box_w = 1.65
    colors = [NAVY, NAVY, TEAL, TEAL, RED, BLUE, ORANGE]
    for index, ((num, title, body), x, color) in enumerate(zip(stages, xs, colors)):
        box = FancyBboxPatch(
            (x, 1.75),
            box_w,
            2.25,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.12, 3.55),
                0.38,
                0.36,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(x + 0.31, 3.73, num, ha="center", va="center", color="white", weight="bold")
        ax.text(
            x + 0.12,
            3.28,
            title,
            ha="left",
            va="top",
            color=NAVY,
            weight="bold",
            fontsize=8.6,
            linespacing=1.05,
        )
        ax.text(x + 0.12, 2.52, body, ha="left", va="top", color="#425466", fontsize=7.4, linespacing=1.30)
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_w + 0.03, 2.88),
                    (xs[index + 1] - 0.05, 2.88),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.1,
                    color="#8FA3B5",
                )
            )
    ax.text(
        0.2,
        5.18,
        "Research architecture: evidence flows forward, governance checks block leakage",
        fontsize=15.5,
        color=NAVY,
        weight="bold",
        va="top",
    )
    ax.text(
        0.2,
        4.78,
        "Gate 7 is presentation-only: no data, feature, model, portfolio, result, or interpretation is changed.",
        fontsize=9.6,
        color="#526579",
        va="top",
    )
    ax.text(
        0.2,
        0.85,
        "Two caught failure modes are part of the evidence: an OOS label boundary omission and a future-label-filtered formation universe.",
        fontsize=9.2,
        color=RED,
        weight="bold",
    )
    ax.text(
        0.2,
        0.45,
        "The correction restored 527 formation rows; all 10 frozen scores were carried into the portfolio layer without winner selection.",
        fontsize=9.2,
        color="#526579",
    )
    save_figure(fig, path)


def figure_rank_ic(root: Path, path: Path, score_order: list[str], labels: dict[str, str]) -> None:
    data = pd.read_csv(root / "reports/tables/gate5_boundary_clean_2014_2025.csv")
    data = data.loc[(data["evidence"] == "rank_ic") & (data["metric"] == "rank_ic")].copy()
    data = data.set_index("signal").loc[score_order].reset_index()
    values = 100 * data["mean"].to_numpy()
    t_stats = data["newey_west_t"].to_numpy()
    colors = [score_color(signal) for signal in data["signal"]]
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    positions = np.arange(len(data))
    bars = ax.bar(positions, values, color=colors, width=0.72)
    ax.axhline(0, color=NAVY, linewidth=0.9)
    ax.set_xticks(positions, [labels[x] for x in data["signal"]], rotation=28, ha="right")
    ax.set_ylabel("Mean daily Spearman rank IC (%)")
    fig.suptitle(
        "Boundary-clean predictive evidence, 2014-2025",
        y=0.985,
        fontsize=15,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        0,
        1.015,
        "Newey-West t-statistics shown above bars; Gate 5 OOS evidence was already observed before Gate 6.",
        transform=ax.transAxes,
        color="#526579",
        va="bottom",
    )
    for bar, value, t_stat in zip(bars, values, t_stats):
        y = value + (0.16 if value >= 0 else -0.20)
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.2f}\nt={t_stat:.2f}", ha="center", va=va, fontsize=7.8)
    ax.set_ylim(min(-1.2, values.min() - 0.8), values.max() + 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99,
        0.96,
        "Gray: base   Teal: Ridge   Blue: tree   Orange: capped gates",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#526579",
    )
    fig.subplots_adjust(top=0.87, bottom=0.25, left=0.08, right=0.98)
    save_figure(fig, path)


def _cost_panel(
    root: Path,
    path: Path,
    neutralization: str,
    title: str,
    score_order: list[str],
    labels: dict[str, str],
) -> None:
    performance = pd.read_csv(root / "reports/tables/gate6_performance_summary.csv")
    costs = pd.read_csv(root / "reports/tables/gate6_cost_stress.csv")
    selector = (
        (performance["period"] == "boundary_clean_primary")
        & (performance["weighting"] == "rank_weight")
        & (performance["neutralization"] == neutralization)
        & (performance["bucket"] == "all")
    )
    gross = performance.loc[selector].set_index("signal").loc[score_order]
    cost_selector = (
        (costs["period"] == "boundary_clean_primary")
        & (costs["weighting"] == "rank_weight")
        & (costs["neutralization"] == neutralization)
        & (costs["bucket"] == "all")
        & (costs["one_way_cost_bps"] == 20)
    )
    stressed = costs.loc[cost_selector].set_index("signal").loc[score_order]
    x = np.arange(len(score_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.8, 6.35))
    gross_values = 100 * gross["annualized_return"].to_numpy()
    stress_values = 100 * stressed["annualized_return"].to_numpy()
    first = ax.bar(x - width / 2, gross_values, width, label="Gross", color=BLUE)
    second = ax.bar(x + width / 2, stress_values, width, label="20 bps one-way stress", color=ORANGE)
    ax.axhline(0, color=NAVY, linewidth=0.9)
    ax.set_xticks(x, [labels[s] for s in score_order], rotation=28, ha="right")
    ax.set_ylabel("Arithmetic annualized return (%)")
    fig.suptitle(title, y=0.985, fontsize=15, weight="bold", color=NAVY)
    ax.legend(loc="upper left", ncol=2)
    ax.text(
        0,
        1.015,
        "Boundary-clean 2014-2025; rank weights; missing-terminal loss100 path; cost is a stress, not calibrated realized cost.",
        transform=ax.transAxes,
        color="#526579",
        va="bottom",
    )
    for bars in (first, second):
        for bar in bars:
            value = bar.get_height()
            offset = 0.13 if value >= 0 else -0.18
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.2,
                rotation=90,
            )
    ax.set_ylim(min(-7.2, stress_values.min() - 1.0), max(11.4, gross_values.max() + 1.2))
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=0.87, bottom=0.27, left=0.08, right=0.98)
    save_figure(fig, path)


def figure_wealth(root: Path, path: Path, labels: dict[str, str]) -> None:
    daily = pd.read_parquet(root / "data/processed/gate6_v1/evidence/daily_portfolios.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.loc[daily["date"] >= pd.Timestamp("2014-01-02")]
    representatives = ["base_family_score", "ridge_explicit_interactions", "tree_alpha_state"]
    line_colors = {representatives[0]: GRAY, representatives[1]: TEAL, representatives[2]: BLUE}
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.9), sharex=True)
    for ax, neutralization, panel in zip(
        axes,
        ["raw", "risk_neutralized"],
        ["A. Raw rank-weight portfolios", "B. Risk-neutralized rank-weight portfolios"],
    ):
        for signal in representatives:
            variant = f"{signal}__rank_weight__{neutralization}__all"
            subset = daily.loc[daily["variant_id"] == variant, ["date", "gross_return_loss100"]].copy()
            subset = subset.sort_values("date")
            subset["wealth"] = (1.0 + subset["gross_return_loss100"]).cumprod()
            ax.plot(subset["date"], subset["wealth"], label=labels[signal], color=line_colors[signal], linewidth=1.6)
        ax.axhline(1.0, color="#9AA7B2", linewidth=0.8)
        ax.set_ylabel("Gross wealth (1.0 start)")
        ax.text(0.01, 0.94, panel, transform=ax.transAxes, va="top", ha="left", color=NAVY, weight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Gross wealth paths for predeclared research roles, 2014-2025",
        y=0.985,
        fontsize=15,
        weight="bold",
        color=NAVY,
    )
    fig.text(
        0.5,
        0.946,
        "Base control, interpretable conditional model, and nonlinear conditional model; role-based display, not winner selection.",
        ha="center",
        color="#526579",
        va="top",
        fontsize=9,
    )
    axes[-1].set_xlabel("Date")
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.08, right=0.98, hspace=0.20)
    save_figure(fig, path)


def figure_exposures(root: Path, path: Path, score_order: list[str], labels: dict[str, str]) -> None:
    data = pd.read_csv(root / "reports/tables/gate6_exposure_summary.csv")
    data = data.loc[(data["weighting"] == "rank_weight") & (data["bucket"] == "all")].copy()
    metrics = [
        ("mean_beta_raw", "Mean beta exposure", "coolwarm", True),
        ("mean_size_z", "Mean size z exposure", "coolwarm", True),
        ("mean_industry_one_way_imbalance", "Industry one-way imbalance", "YlOrRd", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.0), sharey=True)
    for ax, (metric, title, cmap, symmetric) in zip(axes, metrics):
        pivot = data.pivot(index="signal", columns="neutralization", values=metric).loc[score_order]
        pivot = pivot[["raw", "risk_neutralized"]]
        values = pivot.to_numpy()
        if symmetric:
            bound = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-6)
            image = ax.imshow(values, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
        else:
            image = ax.imshow(values, cmap=cmap, vmin=0, vmax=max(np.nanmax(values), 1e-6), aspect="auto")
        ax.set_xticks([0, 1], ["Raw", "Risk-neutralized"], rotation=20, ha="right")
        ax.set_title(title, fontsize=10.5)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                shown = f"{value:.3f}" if abs(value) >= 0.001 else f"{value:.1e}"
                ax.text(col, row, shown, ha="center", va="center", fontsize=7.2, color="#17202A")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    axes[0].set_yticks(np.arange(len(score_order)), [labels[s] for s in score_order])
    axes[0].set_ylabel("Frozen score")
    fig.suptitle("Support-preserving neutralization removes targeted beta, size, and industry tilts", y=0.985, fontsize=14, weight="bold", color=NAVY)
    fig.text(
        0.5,
        0.935,
        "Rank-weight portfolios; prefill constraints. Postfill execution failures remain cash and are not reoptimized.",
        ha="center",
        color="#526579",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.19, right=0.97, wspace=0.30)
    save_figure(fig, path)


def figure_liquidity_capacity(
    root: Path, path: Path, score_order: list[str], labels: dict[str, str]
) -> None:
    liquidity = pd.read_csv(root / "reports/tables/gate6_liquidity_bucket_summary.csv")
    buckets = [
        "adv_low",
        "adv_mid",
        "adv_high",
        "amihud_liquid",
        "amihud_mid",
        "amihud_illiquid",
    ]
    bucket_labels = ["ADV low", "ADV mid", "ADV high", "Amihud liquid", "Amihud mid", "Amihud illiquid"]
    matrix = (
        liquidity.pivot(index="signal", columns="bucket", values="annualized_return")
        .loc[score_order, buckets]
        .to_numpy()
        * 100
    )
    participation = pd.read_csv(root / "reports/tables/gate6_participation_summary.csv")
    participation = participation.loc[
        (participation["weighting"] == "rank_weight")
        & (participation["neutralization"] == "risk_neutralized")
        & (participation["bucket"] == "all")
    ].set_index("signal").loc[score_order]
    p95_adv = 100 * participation["mean_entry_participation_p95_100m"].to_numpy()
    fraction_5 = 100 * participation["mean_entry_fraction_gt_5pct_100m"].to_numpy()

    fig = plt.figure(figsize=(12.8, 9.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.42)
    ax1 = fig.add_subplot(grid[0])
    bound = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    image = ax1.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    ax1.set_xticks(np.arange(len(buckets)), bucket_labels, rotation=18, ha="right")
    ax1.set_yticks(np.arange(len(score_order)), [labels[s] for s in score_order])
    ax1.set_title("A. Raw equal-weight gross returns by formation-date liquidity tercile", loc="left", fontsize=11.2)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax1.text(col, row, f"{value:.1f}%", ha="center", va="center", fontsize=7.1)
    colorbar = fig.colorbar(image, ax=ax1, fraction=0.025, pad=0.02)
    colorbar.set_label("Annualized return (%)")

    ax2 = fig.add_subplot(grid[1])
    x = np.arange(len(score_order))
    width = 0.36
    ax2.bar(x - width / 2, p95_adv, width, label="Mean cohort entry p95 / ADV", color=BLUE)
    ax2.bar(x + width / 2, fraction_5, width, label="Mean fraction of entries > 5% ADV", color=ORANGE)
    ax2.set_xticks(x, [labels[s] for s in score_order], rotation=28, ha="right")
    ax2.set_ylabel("Percent")
    ax2.set_title("B. $100M reference NAV participation diagnostics, risk-neutral rank weights", loc="left", fontsize=11.2)
    ax2.legend(loc="upper left", ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Liquidity and capacity are binding caveats, not an implementation footnote", y=0.985, fontsize=14, weight="bold", color=NAVY)
    fig.text(
        0.5,
        0.947,
        "Participation uses lagged ADV and is diagnostic only; no spread or market-impact model is calibrated.",
        ha="center",
        color="#526579",
        fontsize=9,
    )
    save_figure(fig, path)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))
    from state_alpha.reporting.gate7 import (  # noqa: PLC0415
        FIGURE_FILES,
        SCORE_LABELS,
        SCORE_ORDER,
        sha256_file,
        verify_locked_sources,
    )

    config_path = args.config if args.config.is_absolute() else project_root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    observed = verify_locked_sources(project_root, config)
    output_dir = output_root / config["figures"]["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    figure_pipeline(output_dir / FIGURE_FILES[0])
    figure_rank_ic(project_root, output_dir / FIGURE_FILES[1], SCORE_ORDER, SCORE_LABELS)
    _cost_panel(
        project_root,
        output_dir / FIGURE_FILES[2],
        "raw",
        "Raw rank-weight portfolios: positive gross evidence, material cost sensitivity",
        SCORE_ORDER,
        SCORE_LABELS,
    )
    _cost_panel(
        project_root,
        output_dir / FIGURE_FILES[3],
        "risk_neutralized",
        "Risk-neutral rank weights: cleaner exposures, tighter implementation economics",
        SCORE_ORDER,
        SCORE_LABELS,
    )
    figure_wealth(project_root, output_dir / FIGURE_FILES[4], SCORE_LABELS)
    figure_exposures(project_root, output_dir / FIGURE_FILES[5], SCORE_ORDER, SCORE_LABELS)
    figure_liquidity_capacity(project_root, output_dir / FIGURE_FILES[6], SCORE_ORDER, SCORE_LABELS)

    figure_hashes = {name: sha256_file(output_dir / name) for name in FIGURE_FILES}
    manifest = {
        "schema": "state_alpha_gate7_figure_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "presentation_only": True,
        "config_sha256": sha256_file(config_path),
        "locked_source_hashes": observed,
        "figures": figure_hashes,
    }
    manifest_path = output_dir / "gate7_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"figures": len(figure_hashes), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
