# State-Dependent Machine-Learning Alpha in U.S. Equities

**Dongjun Miao** · CRSP 2000–2025 · [Research audit](docs/backtest_integrity_audit.md)

This research asks when interpretable price-based equity signals work across stock, liquidity, crowding, and market states. Three alpha families (relative-strength breakout, price/trend breakout, oversold recovery) are conditioned on state variables. The public repository packages the existing research implementation and saved aggregate evidence.

## Data and features

The original source is a licensed CRSP CIZ-style daily equity export with returns, return-ex-distribution, OHLC, volume, shares, security history, exchange, and delisting fields. The frozen [feature specification](configs/features.yaml) has **38 features**: 14 alpha signals, 6 stock state, 6 liquidity state, 4 attention/crowding state, 6 market state, and 2 controls. The point-in-time primary universe uses lagged price, market cap, and ADV screens. Licensed data and row-level derivatives are excluded. See [data access](docs/data_access.md).

## Models and evaluation

The implementation includes Ridge explicit alpha-by-state interactions, linear ablations, and constrained shallow histogram trees. Training uses annual expanding refits with label-end purge and a 20-session embargo. 2010–2012 was model-selection validation; 2013 is a transition-sensitivity year because of a discovered validation-boundary omission. **2014–2025 is the boundary-clean comparison.** Gate 5/6 are secondary locked analyses after earlier OOS results were observed.

Gate 6 rescored frozen models on the full point-in-time formation universe. Portfolios form every fifth market session, execute at the next close, and use two fixed half-capital sleeves for 10 subsequent return sessions. Failed execution notional remains cash. All ten scores were carried to predefined portfolio variants.

## Verified aggregate results

From saved [Gate 5](reports/tables/gate5_boundary_clean_2014_2025.csv) and [Gate 6](reports/tables/gate6_performance_summary.csv) summary tables, the boundary-clean tree alpha+state score has mean cross-sectional Rank IC **0.0377** (Newey–West t=5.33); its raw rank-weight long-short portfolio has **9.71% gross annualized**, Sharpe **0.69**, and **34.3 bps** one-way break-even cost. The risk-neutralized counterpart has **7.94% gross annualized**, Sharpe **1.14**. These are historical secondary results, not a live or calibrated net-return claim. [Cost stresses](reports/tables/gate6_cost_stress.csv), [participation diagnostics](reports/tables/gate6_participation_summary.csv), and [provenance](docs/result_provenance.md) are included.

State-only models remain strong controls; tree alpha+state does not decisively beat interaction Ridge on Rank IC. Capped gates did not establish the proposed loss-suppression mechanism. Trading costs and ADV participation materially weaken implementation.

## Reproduce

Python 3.11–3.13. Install from the repository root:

```bash
python -m pip install -e ".[tree,research,dev]"
python -m pytest -q -p no:cacheprovider
python scripts/public_smoke_test.py
python scripts/plot_public_results.py
```

The smoke test uses **synthetic fixtures only**. The plot command reads the included aggregate CSVs. Full empirical pipelines (`scripts/build_canonical.py`, `scripts/build_features.py`, `scripts/run_gate4_baselines.py`, `scripts/run_gate5_secondary_locked.py`, `scripts/run_gate6_portfolios.py`) require authorized CRSP data, frozen intermediate artifacts, and adjusted local paths. The old Gate 7 PDF builder also requires private row-level evidence and is retained as historical source, not as a public-data reproduction command.

## Limits

No independent full-data rerun is possible from this public package. 2013 is transition sensitivity; later gates are not untouched holdouts. Costs are stress scenarios; short borrow, impact, and live slippage are not calibrated. The reference-NAV ADV measure is a participation diagnostic, not capacity. Read the [integrity audit](docs/backtest_integrity_audit.md) before using the numbers.
