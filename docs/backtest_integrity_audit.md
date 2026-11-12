# Backtest integrity audit

## Temporal validity

The original Gate 4 validation boundary allowed 1,065 optimizer-validation labels to reach 2013; 2,965 rows fail the complete label-end purge plus 20-session embargo at that boundary. The original artifact was not retuned. The primary comparison here is 2014–2025; 2013 is transition sensitivity. Annual expanding refits, label-end purging, and a 20-session training embargo are implemented in the source, but this public package cannot independently rerun the licensed-data backtest.

## Formation and labels

Gate 5 prediction rows required valid future labels. Gate 6 corrected the formation universe by rescoring frozen models on the full point-in-time primary universe, restoring 527 rows. Forward labels were excluded from formation scores. Failed t+1-close executions remained cash rather than triggering outcome-aware reranking.

## Data and delistings

The original CIZ `DlyRet` includes calculable delisting returns. Valid terminal returns were applied, then proceeds held in cash. For 703 missing terminal returns, the primary path uses a conservative -100% convention; 0% is a sensitivity bound. These are assumptions, not observed returns. Universe filters are lagged and use common-equity status, price, market cap, and 20-day ADV. Survivorship still cannot be audited from the public aggregates alone.

## Model selection and inference

Gate 5/6 are secondary locked analyses after earlier OOS outcomes were observed. No untouched holdout or winner claim is made. State-only controls explain a material share of rank IC; the tree does not decisively outperform interaction Ridge. Capped gates do not confirm adverse-state suppression. Rank IC uses overlapping 10-session forward labels; the Gate 6 portfolio uses delayed execution and staggered daily P&L. Reported NW statistics and cost stress come from the saved aggregate artifacts.

## Costs and capacity

The 5/10/20/30 bps one-way costs are stress assumptions, not calibrated execution costs. The $100M reference-NAV ADV participation diagnostic is not a capacity estimate. Short borrow, impact, and live slippage remain unmodeled. Public release status is therefore conditional on labeling all returns gross or scenario-based.
