"""Recreate public figures from saved aggregate CSVs only."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
def read(name):
    with (root / "reports" / "tables" / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
ic = read("gate5_boundary_clean_2014_2025.csv")
perf = read("gate6_performance_summary.csv")
cost = read("gate6_cost_stress.csv")
keys = ["ridge_explicit_interactions", "tree_alpha_state", "tree_state_only", "linear_capped_gate", "tree_capped_gate"]
labels = ["Interaction Ridge", "Tree alpha+state", "Tree state-only", "Linear capped", "Tree capped"]
def select(table, score, neutral=None):
    return next(r for r in table if r.get("signal") == score and
                r.get("period", r.get("sample")) in {"boundary_clean_primary", "boundary_clean_2014_2025"} and
                (neutral is None or r.get("neutralization") == neutral) and
                ("weighting" not in r or r["weighting"] == "rank_weight") and
                ("bucket" not in r or r["bucket"] == "all"))
fig, ax = plt.subplots(figsize=(8,4.2))
ax.bar(labels, [float(next(r for r in ic if r["evidence"] == "rank_ic" and r["signal"] == k and r["sample"] == "boundary_clean_2014_2025")["mean"]) for k in keys], color="#336b75")
ax.set_ylabel("Mean daily Rank IC"); ax.set_title("Boundary-clean 2014–2025")
ax.tick_params(axis="x", rotation=22); fig.tight_layout(); fig.savefig(root/"figures/rank_ic.png", dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(8,4.2)); x = range(len(keys))
ax.bar([i-.18 for i in x], [100*float(select(perf,k,"raw")["annualized_return"]) for k in keys], .36, label="Raw", color="#336b75")
ax.bar([i+.18 for i in x], [100*float(select(perf,k,"risk_neutralized")["annualized_return"]) for k in keys], .36, label="Risk-neutralized", color="#b15c48")
ax.set_xticks(list(x), labels, rotation=22); ax.set_ylabel("Gross annualized return (%)"); ax.legend(); fig.tight_layout(); fig.savefig(root/"figures/gross_returns.png",dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,4.2))
for neutral in ["raw", "risk_neutralized"]:
    series = sorted([r for r in cost if r["period"] == "boundary_clean_primary" and r["signal"] == "tree_alpha_state" and r["weighting"] == "rank_weight" and r["neutralization"] == neutral and r["bucket"] == "all"], key=lambda r: float(r["one_way_cost_bps"]))
    ax.plot([float(r["one_way_cost_bps"]) for r in series], [100*float(r["annualized_return"]) for r in series], marker="o", label=neutral)
ax.set(xlabel="Assumed one-way cost (bps)", ylabel="Scenario annualized return (%)", title="Tree alpha+state cost stress, 2014–2025")
ax.legend(); fig.tight_layout(); fig.savefig(root/"figures/cost_stress.png",dpi=160); plt.close(fig)
print("Recreated 3 figures from saved aggregate tables")
