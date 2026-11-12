# Result provenance

These figures are recomputed from saved aggregate CSV rows, not from CRSP or synthetic data. The original Gate 6 validation file reported `passed: true`, 27 output files verified, 1,596,066 formation rows, 652 dates, and 100 variants. Its raw path-bearing JSON is excluded.

| Public aggregate | SHA-256 of original |
|---|---|
| `reports/tables/gate5_boundary_clean_2014_2025.csv` | `4871aa0b16e7b67ce0862b376ecd5378ea278a6059e9ac6f2f10f43649f2d7fe` |
| `reports/tables/gate5_paired_comparisons.csv` | `22375966fb5a281c7288327e91574a86146b8674e5ebdd110f9c42efc5a79851` |
| `reports/tables/gate6_performance_summary.csv` | `319702b55001aa444440c275266496ad7495b310c8061bceeb92d1792b0ed36d` |
| `reports/tables/gate6_cost_stress.csv` | `3558c1c2433267e5566ca13df473af636e422739bcd8117a362e206e6185fe01` |
| `reports/tables/gate6_participation_summary.csv` | `2a9f845027b38e5b72e2bcef272805f2958983b0186e2810077e4125156a9546` |
| `reports/tables/gate6_formation_domain_audit.csv` | `2ea103cf4d97402764a2809da083d43d67b7374f2ea96bb03cec067f6b902691` |

`docs/verified_metrics.json` selects boundary-clean 2014–2025 rows, rank-weight, all-liquidity bucket, with raw and risk-neutralized portfolios. `figures/` contains plots derived from the listed CSVs. Historical handoff prose is not treated as independent evidence.
