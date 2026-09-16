# Feature explorer benchmarks

The harness builds deterministic synthetic data outside the timed region, warms
the requested workload, then records seven end-to-end repetitions, strict output
size/hash, traced allocation, process peak RSS, and environment metadata.

```bash
uv run --group dev python benchmarks/feature_hierarchy_explorer.py \
  --suite acceptance --fixture low6 --workload explore --repeats 7 \
  --output /tmp/bea-explorer-results
```

Run memory-sensitive fixture/workload combinations in separate child processes.
`resource.ru_maxrss` has platform-specific units and is reported without silently
converting them. Its baseline subtraction is a high-water diagnostic, not an
exact allocation measurement; `tracemalloc` is recorded separately.

The selected production S2 backend is typed encoded observed-prefix refinement.
The selection follows the reviewed probes in the implementation plan and is
guarded by differential tests. This durable harness measures complete public
requests; acceptance results remain machine-specific and generated JSON should
be kept outside project/task state.

## September 16, 2026 validation snapshot

On macOS ARM64 with Python 3.13.11, pandas 3.0.5, NumPy 2.5.3, seed 721,
150,000 rows, six four-level string columns, and seven timed repetitions:

| Workload | Median | Incremental peak RSS | Traced peak |
|---|---:|---:|---:|
| bounded `explore()` through strict JSON | 0.192 s | 31.7 MiB | 23.7 MiB |
| S1 top five | 0.013 s | 27.0 MiB | 8.6 MiB |
| two-key S3 | 0.034 s | 27.1 MiB | 18.4 MiB |
| bounded 200-line rendering request | 0.185 s | 22.2 MiB | 23.7 MiB |

RSS figures were converted from macOS byte-valued `ru_maxrss`; the machine JSON
retains platform units. These runs pass the provisional gates for this primary
fixture. They do not substitute for every family in the documented matrix;
high/mixed-cardinality, null-heavy, sparse-domain, over-envelope, and historical-
scale renderer runs remain diagnostics to execute on the designated benchmark
machine before a release claim covering the full matrix.

Three-repeat diagnostic runs on the same machine also passed the bounded combined
gate: the six-by-100-level fixture had a 0.338 s median, 18.1 MiB incremental
peak RSS, and 22.9 MiB traced peak; the mixed-cardinality fixture had a 3.010 s
median, 106.4 MiB incremental peak RSS, 188.6 MiB traced peak, and 19.5 MiB
strict-JSON output. The shorter repeat count makes these diagnostic evidence,
not the final seven-repeat release record.
