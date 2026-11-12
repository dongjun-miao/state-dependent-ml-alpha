# Delisting-return contract

Version: 1.0  
Frozen: 2026-07-31

## Decision

CRSP CIZ `DlyRet` is the daily total-return field used by Project 2. When CRSP
can calculate a delisting return, it is already incorporated in `DlyRet`; a
second standalone `DLRET` must not be joined or compounded again.

Evidence:

- The official CIZ database guide defines `DelRet` as the daily holding-period
  delisting return and states that its storage date is conventionally the
  trading date immediately after the delisting date:
  https://indexes.morningstar.com/docs/guide/crsp-us-stock-databases-guide-for-flat-file-format-2-0
- The WRDS CIZ event-study implementation uses `DlyRet` directly and notes that
  the delisting return is already incorporated:
  https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/run-an-event-study-ciz-format-macro/
- CRSP's official return-missing flag appendix defines the observed missing
  reasons:
  https://www.crsp.org/wp-content/uploads/appendix/FlagType_RM.html

## Observed Project 2 data

The canonical panel contains 14,612 unique delisting security-days:

| Return state | Rows | Interpretation |
|---|---:|---|
| Numeric `DlyRet` | 13,909 | CRSP-observed total return, including calculable delisting effect |
| `DG` | 358 | Delisting price more than ten periods after delisting date |
| `DM` | 197 | Delisting price/amount missing |
| `DP` | 148 | Delisting pending |
| **Missing total** | **703** | Unknown terminal economic value |

The export's `DelActionType`, `DelStatusType`, `DelReasonType`, and
`DelPaymentType` columns are null on all 14,612 canonical delisting days.
Consequently, Project 2 cannot justify a reason-dependent empirical imputation
from this extract.

## Label treatment

For a valid terminal return inside a label window:

1. compound CRSP daily total returns through the `DlyDelFlg == Y` row;
2. treat the security as liquidated into cash at that terminal value;
3. hold cash through the label's fixed calendar end;
4. compound the market benchmark through the full fixed horizon.

For a missing terminal return:

- primary label: missing;
- lower terminal-value bound: −100%;
- upper terminal-value bound: 0%;
- both bounds preserve the observed pre-terminal path.

The 0% terminal value is an explicit sensitivity boundary, not an observed
return, a point estimate, or permission to fill raw CRSP data with zero.

## Research guardrails

- Never discard a valid delisting row.
- Never add a legacy `DLRET` to CIZ `DlyRet`.
- Never call either sensitivity bound a realised return.
- Report primary-label exclusions and both bounds by horizon.
- Any portfolio claim exposed to an unknown terminal value must survive the
  lower-bound sensitivity or be qualified explicitly.
- The raw and canonical data remain unchanged; treatments exist only in the
  derived Phase 2B label panel.
