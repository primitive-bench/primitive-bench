# bench-stats

Statistics library for Primitive Bench: Wilson intervals, McNemar's paired test,
seeded bootstrap CIs, BEIR/MTEB IR metrics, and the richer reporting helpers
(McNemar power/sizing, Cochran–Mantel–Haenszel omnibus, CUSUM drift detection,
tied-rank winner bands).

The `proportions`, `resampling`, and `retrieval` modules are dependency-light
(stdlib `math`/`random` only). The `reporting` module depends on scipy/statsmodels.

Provenance: the reporting helpers are ported from
[arlenk2021/GoldenEvalsWebSearch](https://github.com/arlenk2021/GoldenEvalsWebSearch)
(Apache-2.0 code), relicensed to MIT within primitive-bench.
