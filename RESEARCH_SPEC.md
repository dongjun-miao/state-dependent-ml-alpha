# Research specification v0.3

Title: **Learning When Technical Signals Work: State-Dependent
Cross-Sectional Alpha Selection in U.S. Equities**

Status: universe, label, price-representation, and 38-feature contracts frozen
through Gate 3; no label correlation, model training, portfolio backtest, or
signal-performance claim has been made.

## 1. Research question and contribution

Do the predictive effects of continuous relative-strength breakout,
price/trend breakout, and oversold-recovery signals depend systematically on
stock, liquidity, attention/crowding, and market states? Can an interpretable
conditional model improve strict out-of-sample rank IC and implementable
long-short performance primarily by suppressing signals in adverse states?

The contribution is not a catalogue of technical indicators and not “LightGBM
predicts returns.” It is a nested test of conditional signal efficacy:

1. define a small, economically motivated set of price-based base alphas;
2. define states separately from alphas;
3. estimate explicit signal × state interactions;
4. test whether nonlinear interactions add to a linear model;
5. identify whether improvement is loss avoidance in bad states;
6. validate under a locked, purged, expanding-window OOS design.

## 2. Falsifiable hypotheses

- **H1 — conditional efficacy:** at least one base-alpha family has stable
  cross-sectional efficacy differences across predeclared states.
- **H2 — gating:** a signal/state model improves OOS rank IC and portfolio
  diagnostics relative to the same base signals without states.
- **H3 — suppression mechanism:** most economically meaningful incremental
  performance comes from reducing exposure when predicted efficacy is poor,
  rather than mechanically increasing exposure everywhere.
- **H4 — nonlinear increment:** a shallow tree model adds information beyond
  explicit regularized linear interactions. Failure to beat the linear model
  means the interaction story may be linear; it is not a failed project.

Negative controls:

- a state-only model tests whether results are generic state prediction rather
  than conditional technical efficacy;
- a base-signal-only model isolates the value of conditioning;
- date-preserving label permutations test pipeline leakage/overfit behavior;
- a model without volume-derived state variables tests whether the narrative
  accidentally becomes another volume-alpha project.

## 3. Data and audit-driven decisions

### 3.1 Source

Shared, read-only CRSP CIZ-style daily exports at
the local licensed-data directory, covering 2000-01-03 through 2025-12-31.
External paths appear only in `configs/paths.yaml`.

The raw schema has price, return, volume, shares, OHLC, quote, security,
delisting-metadata, and embedded market-return fields. Legacy `SHRCD`,
`EXCHCD`, and standalone `DLRET` are absent. Project 2 will use a documented
CIZ data contract and will not synthesize legacy names.

### 3.2 Source selection

The existing 27-column `research_v1` CSV is acceptable for exploratory
distribution checks only. Production labels require a new Project 2 canonical
intermediate retaining at minimum:

- `PERMNO`, `PERMCO`, `DlyCalDt`;
- security effective dates and type/exchange/status fields;
- `DlyPrc`, OHLC, price/return flags, `DlyRet`, `DlyRetx`;
- `DlyDelFlg` and relevant `Del*` metadata;
- `DlyVol`, `ShrOut`, `DlyCap`, adjustment factors;
- embedded market returns.

The intermediate is written to Project 2, never shared raw storage.

### 3.3 Corporate actions

Raw price/OHLC levels cannot be used across splits without adjustment. The
development-only Gate 3 audit found 100% `DlyRetx` and price-factor coverage
but eight adjacent raw-price/factor identity outliers above `1e-4`.
Consequently, every primary rolling price-path feature compounds CIZ
`DlyRetx`. Raw OHLC is reserved for a separately adjusted robustness
specification. No future adjustment factor normalizes historical prices.

### 3.4 OHLC and execution decision

OHLC existence is not enough; coverage, ordering, flags, and corporate-action
continuity must pass validation. Although complete raw OHLC coverage was
99.9021% in the development primary universe, raw-level continuity was not
adopted. The documented `DlyRetx` close-only family in Section 5 is primary.

The primary implementation is deliberately conservative:

- features are finalized after date t close;
- the earliest baseline trade is date t+1 close;
- label P&L begins after that execution.

Open-based execution is, at most, a separately audited robustness check. The
project will not claim next-open execution merely because `DlyOpen` is present.

### 3.5 Delisting return decision

The CIZ export has no standalone `DLRET` because CIZ `DlyRet` already includes
a calculable delisting return. Project 2 compounds through a valid
`DlyDelFlg == Y` row, liquidates to cash, and holds cash through the fixed
label end. Of 14,612 canonical terminal rows, 703 have missing `DlyRet`: 358
`DG`, 197 `DM`, and 148 `DP`. The primary label remains missing for those
unknown values; explicit terminal-value bounds of −100% and 0% are stored for
sensitivity. Zero is not treated as observed or imputed CRSP data. See
`DELISTING_RETURN_CONTRACT.md`.

## 4. Point-in-time universe

The frozen primary universe requires:

- active formation-date close on primary exchange `N`, `Q`, or `A`;
- prior-session CIZ common-equity status `EQTY/COM/NS`;
- prior-session `USIncFlg == Y`;
- lagged absolute price at least $3;
- lagged market capitalization at least $100 million;
- 20-session ADV through t−1 at least $1 million;
- the lag and all ADV observations are consecutive market sessions;
- historical exits retained;
- at least 252 prior security observations.

Ten predeclared absolute/percentile and incorporation variants were compared
using 2000–2012 distributions only. The selected rule retained 7,020,368 of
14,891,569 development analysis-base rows, averaged 2,350.6 names at the
annual median date, and had 9.27% annual-median breadth CV. No signal,
forward-return, IC, portfolio, or 2013–2025 outcome entered the decision. See
`reports/memo/PHASE2B_UNIVERSE_FREEZE.md`.

Multiple share classes remain separate eligible PERMNOs in the baseline, with
PERMCO concentration reported. A robustness sample keeps only the share class
with the largest lagged market cap per PERMCO.

## 5. Base alpha families

Every alpha and state is continuous and built only from information through
t. The only categorical fields are the two explicit controls. Exact ordered
definitions and lineage are frozen in `configs/features.yaml`.

### 5.1 Relative-strength breakout

| Feature | Economic interpretation | Role |
|---|---|---|
| `rs_excess_return_{20,60,126}d` | stock gross return relative to value-weighted market gross return | alpha |
| `rs_percentile_60d` | same-date leadership inside the frozen primary universe | alpha |
| `rs_acceleration_20_minus_60d` | short-minus-medium relative strength | alpha |

### 5.2 Price/trend breakout

| Feature | Economic interpretation | Role |
|---|---|---|
| `price_distance_prior_high_126d` | DlyRetx price index relative to the maximum through t−1 | alpha |
| `price_distance_ma_20d` | location relative to a 20-session arithmetic price-index average | alpha |
| `trend_distance_vol_normalized_60d` | 60-session trend distance on a comparable risk scale | alpha |

The prior 126-session maximum excludes date t. All three features use the
`DlyRetx` chain, not simulated or unadjusted high/low data.

### 5.3 Oversold recovery

| Feature | Economic interpretation | Role |
|---|---|---|
| `reversal_5d` | short-horizon overreaction | alpha |
| `drawdown_{20,60}d` | depth below local DlyRetx-chain highs | alpha |
| `distance_52w_high` | longer-horizon damage | alpha |
| `momentum_slope_reversal` | 5-session return relative to scaled 20-session slope | alpha |
| `recovery_above_ma_10d` | continuous recovery confirmation | alpha |

Every frozen column has exactly one role and appears once in the modeling
matrix.

## 6. State, control, and implementation variables

### 6.1 Stock state

`log_lagged_market_cap`, `beta_252d`, 20/60-session realized volatility,
60-session idiosyncratic volatility, and `momentum_126_21d`.

### 6.2 Liquidity state

`log_adv20_usd`, 20-session turnover, 20-session Amihud illiquidity,
20-session zero-return frequency, log price, and the valid positive
ask-minus-bid midpoint spread. Invalid quotes remain missing.

### 6.3 Attention/crowding state

Abnormal current turnover relative to t−20 through t−1, the current absolute
return scaled by prior 60-session volatility, the 20-session maximum daily
return, and 20-versus-60-session volatility expansion.

Volume and turnover here are conditioning/liquidity/crowding measures. They are
never Project 2's primary alpha.

### 6.4 Market state

20/60-session value-weighted market return, 20-session market realized
volatility, 126-session market drawdown, same-date frozen-primary return
dispersion, and 20-session frozen-primary breadth. Binary regimes are not Gate
3 columns; later interpretation layers must use trailing or training-period
thresholds.

### 6.5 Controls and cost variables

The two categorical controls are two-digit SIC and primary exchange. Size,
beta, ADV, price, turnover, and spread remain single state columns that may be
reused for implementation diagnostics without duplicating them in the feature
matrix. State-only prediction is always reported separately.

## 7. Labels and timing

Primary label:

`fwd_total_return_rank_10d`, the centered date-level rank of the stock's
10-session forward total return inside the frozen formation universe. Because
the fixed-horizon market return is common within date, raw- and
market-excess-return ranks are identical. A residual-return rank remains a
predeclared Gate 4 robustness label using the frozen Gate 3 controls; its
residualization must be fitted within training data only.

Timeline:

```text
t close             t+1 close                    t+11 close
features complete -> baseline execution --------> 10-session label ends
                       label begins after execution
```

Robustness labels are 5-, 10-, and 20-session forward total and market-excess
returns, the two missing-terminal-value bounds, and a later 20-session
residual-return rank.

Rules:

- total returns include validated delisting effects;
- no date-t return is included in a label predicted with date-t features;
- any later market/exposure residualization uses a frozen, point-in-time method;
- overlapping labels are not treated as independent observations;
- the longest 20-day label determines purge and embargo length;
- no forward return, future universe status, future ADV, or full-sample
  statistic enters features or eligibility.

## 8. Evidence ladder and models

Evidence proceeds from simple to complex:

1. daily Pearson and Spearman rank IC for each primitive alpha;
2. quintile/decile sorts and monotonicity;
3. state-conditioned double sorts;
4. daily Fama–MacBeth regressions with HAC inference over coefficient series;
5. Ridge and Elastic Net with explicit alpha × state interactions;
6. shallow LightGBM or preregistered histogram-gradient-boosting fallback;
7. family, state, volume-state, and complexity ablations.

Nested predictive models:

- base signals only;
- states only;
- base signals plus state main effects;
- regularized explicit interactions;
- shallow nonlinear base-signal-only;
- shallow nonlinear signal-plus-state.

This nesting distinguishes conditional efficacy from generic state prediction.
Tree depth/leaves are constrained; hyperparameter grids are in
`configs/models.yaml`.

## 9. OOS validation

Observed data coverage supports:

- initial training: 2001-01-02 to 2009-12-31;
- model-selection validation: 2010-01-04 to 2012-12-31;
- strict OOS: 2013-01-02 to 2025-12-31;
- annual expanding-window refits using data available through the prior year.

The first eligible formation date is later than 2000 because of the 252-day
history requirement.

Rules:

- date-block splits only; never random stock-date splits;
- purge observations whose forward labels overlap the next fold;
- 20-trading-day embargo, matching the longest robustness label;
- choose universe rule, feature families, model class, hyperparameter grid, and
  selection metric using the development period;
- freeze the choice before strict OOS;
- annual refits reuse the frozen rule and may not inspect future OOS years;
- winsorization, normalization, and imputation fit only on training data or use
  an explicitly same-date cross-sectional transform.

## 10. Portfolio translation

Primary portfolio diagnostics:

- weekly ranked long-short formation to reduce mechanical turnover;
- top and bottom deciles, equal-weight and rank-weight variants;
- one-day delayed close execution;
- overlapping 10-day forecasts translated into staggered daily cohorts before
  reporting annualized P&L metrics;
- raw and beta-neutral results;
- position and ADV participation diagnostics.

An average overlapping 10-day forward spread is a forecast diagnostic, not a
daily portfolio return and will never be annualized as one.

Transaction-cost analysis:

- gross performance;
- 5/10/20/30 bps one-way cost stress tests;
- break-even one-way cost;
- turnover and results by ADV/liquidity bucket.

Without a calibrated spread/impact model, these remain cost stress tests, not
realized transaction costs or precise capacity estimates.

## 11. Evaluation

- daily Pearson IC, Spearman rank IC, mean IC, ICIR;
- Newey–West t-stat with lag at least 20 for overlapping horizons;
- decile monotonicity and long-short spread;
- gross return, turnover, annualized volatility, Sharpe, maximum drawdown;
- beta, size, industry, and liquidity exposures;
- year, subperiod, and regime performance;
- break-even and 5/10/20/30 bps stress;
- ADV/liquidity-bucket performance;
- feature-family and state-family ablation;
- linear versus nonlinear incremental performance;
- state-only and volume-state-removal negative controls.

Cross-sectional dependence is handled by daily aggregation/Fama–MacBeth rather
than treating millions of stock-days as independent.

## 12. “Suppress bad signals” mechanism test

For each base family:

1. sort observations on base-signal strength;
2. within strength buckets, sort on predicted conditional efficacy;
3. compare raw exposure with gated exposure;
4. attribute incremental OOS return to:
   - adverse-state loss avoided;
   - favorable-state gain preserved;
   - favorable-state gain amplified;
   - changed turnover and systematic exposures.

Report:

- conditional IC surfaces;
- fraction of raw high-conviction signals suppressed;
- OOS contribution from suppressed versus retained observations;
- performance of a capped gate that can only reduce, not amplify, exposure;
- performance of an unrestricted conditional score.

H3 is supported only if loss avoidance remains after exposure and turnover
controls. If gains come from a standalone market/liquidity timing model, the
claim is rejected.

## 13. Research-integrity controls

| Risk | Required control |
|---|---|
| Look-ahead | lag eligibility; exclude current return from label; tests on window endpoints |
| Survivorship | retain historical exits; compare end dates and delisting flags |
| Delisting bias | reconcile CIZ returns/flags; never fill delisting outcomes with zero |
| Overlap | purge, embargo, HAC, staggered daily P&L |
| Stale prices | zero-return frequency, volume/quote checks, liquidity screens |
| Extreme returns | flag-aware diagnostics; training-only/same-day treatment |
| Corporate actions | validated adjustment factors or return index |
| Volume conventions | venue/time diagnostics; no ad hoc retrospective scaling |
| Multiple classes | PERMCO diagnostics and largest-class robustness |
| Universe leakage | lagged, point-in-time filters only |
| Preprocessing leakage | train-only or same-day transforms |
| Data snooping | feature budget, family preregistration, frozen OOS |
| Repeated OOS selection | final OOS used once under frozen specification |

## 14. Decision criteria and honest outcomes

The project can conclude any of the following:

- state conditioning adds stable, implementable OOS value;
- interactions exist but are adequately linear;
- conditional prediction exists but turnover/cost makes it non-tradable;
- state-only effects explain the result, rejecting the conditional-signal story;
- base technical signals have no robust OOS efficacy in this universe.

No year may be removed for poor performance unless a predeclared, independently
verified data error is documented. In-sample and validation results are labeled
as such. Volume proxies are never described as true order flow, and statistical
significance is not equated with tradability.

## 15. Phase gate

Gates 1–6 were completed under explicit approvals. Gate 4 used the frozen
2010–2012 validation and 2013–2025 annual expanding-window OOS contract.
Gate 5 added the boundary correction, constrained tree, grouped ablations, and
mechanism diagnostics. Gate 6 translated every frozen score into delayed,
staggered portfolio evidence without OOS winner selection. The planned
empirical pipeline is complete.

## 16. Gate 4 locked findings

Gate 4 was published and independently validated on 2026-07-31:

- the unconditional three-family score had strict-OOS mean daily rank IC
  −0.00529 (NW t=−0.99);
- Ridge with frozen explicit interactions had 0.03279 (NW t=5.06);
- Elastic Net with frozen explicit interactions had 0.03502 (NW t=4.69);
- Ridge and Elastic Net IC was positive in 12 of 13 OOS years;
- the strongest Fama–MacBeth interaction was oversold-recovery ×
  realized-volatility, with a negative OOS coefficient (NW t=−4.15);
- Ridge and Elastic Net decile spreads were positive, but their overlapping
  10-session NW t-statistics were only 1.81 and 1.52;
- model-decile membership turnover averaged about 20.7%–20.9% one-way.

Interpretation is deliberately limited. Gate 4 shows that a regularized
conditional ranking function is predictive out of sample. Because the linear
design includes alpha, state, control, and interaction terms together, it does
not by itself identify interactions as the sole source of improvement.
Nested state-only/main-effect/interaction ablations are required in Gate 5.
No annualized P&L, Sharpe, drawdown, transaction-cost, or capacity claim may be
made from the overlapping-label spreads.

### Gate 4 boundary correction

Before Gate 5 fit, an additional validation-to-OOS audit found 1,065 Gate 4
optimizer-validation labels whose endpoints reached 2013. Applying both
label-end purging and the 20-session embargo removes 2,965 validation rows.
Accordingly:

- Gate 4's 2013 result is a transition sensitivity;
- 2014–2025 is the boundary-clean period for reused Gate 4 scores;
- the Gate 4 artifact and parameters are not overwritten or retuned;
- Gate 5 model selection applies the corrected boundary rule.

## 17. Gate 5 secondary locked findings

Gate 5 was frozen after Gate 4 outcomes were known and cannot be described as
a new untouched confirmatory test. It used a clean 70,954-row validation
selector and one common constrained tree complexity.

Boundary-clean 2014–2025 results:

- Ridge interaction rank IC: 0.03611 (NW t=5.32);
- tree alpha+state rank IC: 0.03772 (NW t=5.33);
- tree alpha+state minus alpha-only IC: 0.00729 (NW t=2.13);
- tree alpha+state minus state-only IC: 0.00383 (NW t=1.52);
- Ridge interactions minus main effects IC: 0.00411 (NW t=1.54);
- Ridge interactions minus main effects decile spread: 0.00214
  (NW t=2.55);
- tree alpha+state minus Ridge interactions IC: 0.00161 (NW t=0.50).

These results support conditional ranking but reject a simplistic claim that
nonlinearity or explicit interactions explain the whole effect. State-only
prediction is strong, and the constrained tree does not decisively outperform
regularized linear interactions on IC.

The capped gates suppressed approximately half of high-conviction base
signals. Their boundary-clean net suppression effects had NW t-statistics of
only 1.44 and 1.22. The “suppress bad signals” mechanism is therefore
suggestive, not confirmed.

One implementation detail is retained rather than retrospectively changed:
the frozen deterministic stride cap can reduce the tree optimizer sample when
the uncapped sample first crosses 300,000 rows. Full evaluation is unaffected,
but this non-monotone effective training count must remain a documented
limitation; it may not be redesigned after seeing Gate 5 OOS.

## 18. Gate 6 portfolio findings

Gate 6 was frozen after Gate 5 outcomes and remains secondary locked evidence.
Before P&L, the Gate 5 label-valid prediction table was identified as an
invalid formation universe because forward label availability entered row
selection. Gate 6 therefore:

- recomputed same-date alpha-family ranks on the complete point-in-time
  primary universe;
- applied the frozen annual Gate 4 and Gate 5 model states to 1,596,066 rows
  over 652 every-fifth-session formations;
- restored 527 formation rows excluded by label availability;
- excluded every forward-label column from formation scores;
- used execution validity only after ranking, leaving failed-order notional
  in zero-return cash without redistribution.

All ten scores entered all predeclared portfolio variants. The main
implementation uses formation information through close t, execution at close
t+1, returns from t+2 through t+11, two fixed half-capital cohorts, and
security-level netting of scheduled entries and exits.

Boundary-clean 2014–2025 rank-weight findings:

- raw tree alpha+state: 9.71% annualized gross return, 14.15% volatility,
  Sharpe 0.69, NW t=2.42, 34.3 bps break-even one-way cost;
- risk-neutralized tree alpha+state: 7.94% gross, 6.94% volatility, Sharpe
  1.14, NW t=3.82, 22.4 bps break-even cost;
- raw Ridge interactions: 7.45% gross, Sharpe 0.60, NW t=2.12, 32.8 bps
  break-even cost;
- risk-neutralized Ridge interactions: 4.76% gross, Sharpe 0.76, NW t=2.63,
  16.1 bps break-even cost;
- raw and neutralized unconditional base-family portfolios remained negative.

At a 20 bps one-way stress, raw and risk-neutralized tree alpha+state
annualized returns fell to 4.04% and 0.86%. The cost stress is not a calibrated
realized cost. The $100M reference-NAV diagnostic places average entry p95 ADV
participation near 7% for the neutralized tree alpha+state portfolio, which is
a material warning and not a capacity estimate.

The capped gates did not validate the original mechanism. Their rank-weight
gross returns were only 1.75%–3.20%, with 4.5–9.8 bps break-even costs.
Combined with Gate 5 suppression attribution, H3 remains unconfirmed.

The conservative −100% and 0% missing-terminal conventions differ by at most
0.69% annualized across boundary-clean variants. Valid delisting returns are
applied before terminal proceeds move to cash.

The final defensible conclusion is that conditional alpha+state ranking
produces positive gross portfolio evidence, including after beta, size, and
industry neutralization, but tradability is cost- and capacity-sensitive.
State-only controls remain economically meaningful, nonlinear IC dominance
was not established, and no single Gate 6 variant is selected as an OOS
winner.

## 19. Gate 7 presentation freeze

Gate 7 is a presentation-only release. It adds no data, feature, label, model,
portfolio rule, empirical result, or revised interpretation. Its 13 source
files are SHA-256 locked in `configs/gate7.yaml` before figure or report
generation.

The presentation package contains exactly seven audited figures, a compact
research PDF, a one-page interview brief, and a reproducibility command sheet.
Role-based displays may show the unconditional base control, Ridge explicit
interactions, and tree alpha+state together, but this is not OOS winner
selection. All ten frozen scores remain visible in the principal IC, cost,
exposure, liquidity, and participation displays.

The presentation validator must confirm unchanged source hashes, seven figure
hashes and minimum resolutions, an extractable and structurally complete PDF,
the companion documents, and the still-passed Gate 6 validation report. Any
empirical extension requires a new approval gate and a newly frozen research
contract.
