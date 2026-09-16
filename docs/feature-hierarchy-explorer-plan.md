# Feature Hierarchy Explorer — Implementation and Validation Plan

## 1. Status, scope, and authority

**Planning baseline:** September 15, 2026, repository HEAD `890e08ad0fc18cd951e44fd084579882a4d150d6`. Beatrice approved the reviewed hybrid architecture and requested this plan and the updated Cortex specification. This document does not authorize implementation, a release, a push, or unrelated repairs. No explorer implementation or acceptance results exist yet.

The behavioral source of truth is Cortex:

`projects/personal-projects/bea-tools/feature-hierarchy-explorer-spec.md`, **spec revision 0.2**.

At planning time its local path is `/Users/beatrice/AgentFiles/vaults/Cortex/projects/personal-projects/bea-tools/feature-hierarchy-explorer-spec.md`. Resolve the user's Cortex checkout if that machine-specific path changes. Read the specification before implementation; this document references its sections rather than maintaining another copy of its contracts. If a contract must change, reconcile the source specification before implementing the change.

This repository document owns code sequencing, file responsibilities, tests, and benchmark procedure. Cortex remains the authority for project/task state. The steps below are delivery phases, not a second task ledger.

### Approved design

- Pandas public input; standard-library result types; direct NumPy dependency for encoded kernels.
- Independent S1 counting fast path; encoded observed-prefix refinement as the leading S2 candidate; normalized pandas FD baseline for S3.
- Compact parent-linked nodes and typed level dictionaries instead of repeated full paths.
- Population-preserving bounded expansion by default; explicit conditional pre filtering.
- Independent oracle and end-to-end benchmark gates before backend selection.
- Separate inference proposals from chosen dimensions and observed dependencies.

**Production backend selection remains benchmark-gated.** The approved direction is not an instruction to force every operation through NumPy or to implement every experimental backend as maintained production code.

## 2. Verified repository baseline and integration risks

The tracked repository is small and has no existing `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, explorer package, documentation directory, or test workflow at the baseline. Recheck instructions and status before changing it; preserve unrelated work.

| Existing surface | Verified condition | Planned handling |
|---|---|---|
| `pyproject.toml` | Python ≥3.10; pandas ≥2.2; pydicom ≥3.0; setuptools/setuptools-scm build | Align Python/pandas floors; declare NumPy and dev/optional dependencies; retain unrelated pydicom policy |
| `bea_tools/__init__.py` | Eager sampler and plotting exports | Preserve exported names via lazy optional-module boundaries; eagerly register only lightweight accessors/core API |
| `bea_tools/_pandas/__init__.py` | Imports sampler; does not import `series` | Remove import-order dependency and register existing Series accessor |
| `bea_tools/_pandas/sampler.py:6` | Eager `pulp` import | Isolate behind sampling extra and actionable lazy-boundary error |
| `bea_tools/_matplotlib/general.py:4` | Eager matplotlib import and pyplot annotations | Isolate behind plotting extra; handle annotations if imports move inside functions |
| `bea_tools/testing/test_lp_sampler_weight_normalization.py` | Three existing pytest tests, requiring PuLP/CBC | Preserve and run in extras-enabled regression job |
| `bea_tools/testing/_pandas.py` | Legacy imports reference absent `Level`, `Feature`, `SamplingNode`, `TreeSampler` | Document as legacy; do not silently widen discovery or repair wholesale |
| `.github/workflows/publish.yml` | Release publication only | Add independent tests workflow; leave publication behavior unchanged |
| `README.md` | Minimal installation/requirements documentation | Add supported versions, extras, entry-point examples, and plan/API links |

The existing `/tmp/bea-clean` environment reproduced `ModuleNotFoundError: No module named 'pulp'` on package import, with Python 3.13.14, pandas 3.0.5, and NumPy 2.5.3. Pandas metadata requires Python ≥3.11. The review did not install packages or rerun the sampler suite; the first draft's reported three passes are historical, not a current test result.

## 3. Planned file map

Paths below are proposed additions unless identified as existing. Favor these responsibility boundaries over rigid line counts; combine small private helpers when that improves clarity without mixing contracts.

| Path | Responsibility |
|---|---|
| `bea_tools/_explore/__init__.py` | Public explorer exports and lightweight registration wiring |
| `bea_tools/_explore/encoding.py` | Validation, canonical typed identities, NA normalization, per-column codes/dictionaries, overflow-safe grouping inputs |
| `bea_tools/_explore/result.py` | Dataclasses, scope/population records, compact nodes, typed scalar JSON, canonical output |
| `bea_tools/_explore/census.py` | `levels`, `census`, bounded selection, pre/post scopes, `explore` orchestration |
| `bea_tools/_explore/_kernels.py` | Private pandas reference and selected encoded grouping kernels |
| `bea_tools/_explore/grain.py` | `KeySpec`, composite key groups, FD statistics, observational determinant relations |
| `bea_tools/_explore/relations.py` | Sparse pair counts, Cramér's V, reference-domain reconciliation, absence totals/examples |
| `bea_tools/_explore/roles.py` | Evidence-backed `SchemaProposal`; no silent selection or confidence score |
| `bea_tools/_explore/render.py` | Result-only bounded renderer, control escaping, safe/native Unicode modes |
| `bea_tools/_explore/accessor.py` | Thin `DataFrame.bea` delegation with no alternate engine |
| `bea_tools/testing/explore/conftest.py` | Deterministic shared synthetic fixtures |
| `bea_tools/testing/explore/oracle.py` | Tiny-frame independent Python-loop counting/FD/absence oracle |
| `bea_tools/testing/explore/test_*.py` | Contract, differential, metamorphic, API, and renderer suites |
| `bea_tools/testing/test_explore_imports.py` | Fresh-process core/optional import and accessor regressions |
| `benchmarks/feature_hierarchy_explorer.py` | Reproducible synthetic fixture generation, interchangeable candidate adapters, subprocess timings/memory |
| `benchmarks/README.md` | Benchmark protocol, hardware scope, result interpretation, reproduction commands |
| `docs/feature-hierarchy-explorer.md` | User guide and worked API examples after implementation |
| `.github/workflows/tests.yml` | Core, extras, supported-version, wheel-smoke, lint, and build checks |

Modify existing `pyproject.toml`, package initializers, optional import boundaries, and `README.md` only as needed. Avoid sampler algorithm changes, broad utility refactors, or edits to the publishing workflow in this increment.

## 4. Delivery phases and exit criteria

### P0 — Contract fixtures and reproducible foundation

**Depends on:** reading spec revision 0.2 and rechecking repository state.

1. Turn spec §§4–9 into literal contract fixtures before optimized implementation: typed values, population accounting, compact node records, key syntax, warning/error codes, and canonical ordering.
2. Define the scalar codec precisely for supported temporal types/resolutions, aware-time UTC normalization, non-finite floats, large integer labels, and nested tuple column labels. Unsupported objects/ranges fail explicitly. These are conformance details under the typed identity policy, not permission to replace it with pandas' mixed-number equality.
3. Freeze the result schema version `0.2`; implement standard-library dataclasses and strict `to_dict()` output. Add literal serializer tests that do not depend on production grouping.
4. Plan supported versions as Python ≥3.11 and pandas ≥3.0. Start dependency-floor verification with NumPy ≥1.26 on Python 3.11; test actual APIs before freezing the NumPy floor. Respect newer Python-specific NumPy requirements from pandas rather than imposing one incompatible old pin across the matrix.
5. Add a `dev` dependency group with pytest, Hypothesis (or an equivalently exhaustive seeded generator), Ruff, and build tooling. Define extras `sampling` (PuLP), `plotting` (matplotlib), and `unicode` (wcwidth). Keep core explorer imports independent of all extras.
6. Establish import-smoke tests, then fix lazy optional boundaries and accessor registration. Catch only the relevant missing optional dependency; preserve unrelated import errors. Preserve `from bea_tools import LPSampler` and existing plotting names in their installed-extra environments.
7. Add public stubs only as part of the test-first implementation sequence; a committed phase is complete only when its promised behavior is exercised, not merely importable.

**Exit:** documented version/dependency matrix, literal codec/result tests, minimal installed-package import success without PuLP/matplotlib, both accessor registrations, extras import success, and unchanged existing export behavior. This phase fixes packaging, not the sampler algorithm.

### P1 — Independent oracle and pandas reference

**Depends on:** P0 identity and result contracts.

1. Write a small Python-loop oracle using independent typed tokens and explicit grouping, with expected literal fixtures. It must not call production encoding, pandas groupby, or the production result builder to decide expected values.
2. Cover S1 counts, all observed prefixes, composite FD groups, support counts, pair mappings, and small dense absence/association tables.
3. Implement normalized pandas S1 and prefix grouping as the reference, explicitly controlling `observed`, missingness, and intermediate sorting.
4. Implement separate `levels()` and `census()` results; keep `infer_schema` outside the analysis path. Do not optimize until oracle parity passes.

**Exit:** literal and generated tiny frames agree with the independent oracle; basic all-null/empty/mixed-type/composite-key cases are already represented in tests.

### P2 — Encoding reuse and candidate S2 kernel

**Depends on:** P1 reference parity.

1. Factorize needed columns once per compatible call scope, with exact typed normalization for mixed families and one NA identity. Homogeneous fast paths must pass the same contract tests.
2. Implement exact observed `(parent_id, level_code)` refinement and dense integer counting. Retain representative indexes/links without materializing a Python path tuple for every row.
3. Widen before packed-key multiplication, verify range before casting, and exercise an exact-pair fallback. Test the fallback with a deliberately lowered internal packing threshold or direct boundary input; a huge allocation is not a useful overflow test.
4. Keep code IDs private and separate from canonical level/node IDs. Canonicalize published output independently of factorization encounter order.
5. Implement compact parent-linked nodes and referenced-level dictionaries. Build dictionary entries only for output-referenced levels, required retained-set metadata, and bounded examples; count omitted cardinality without serializing every value.
6. Add benchmark-only adapters for leaf rollup, encoded pandas grouping, lexsort prefixes, and repeated `unique(axis=0)`. Reuse one output adapter so end-to-end comparisons produce identical strict JSON. Experimental adapters need not become supported production backends.
7. Share encoding within `explore()` only when identity, rows, and column scope are compatible. Avoid persistent accessor caches on mutable caller dataframes.

**Exit:** each candidate matches the oracle/reference on supported tiny/random inputs; the leading kernel handles missingness, categorical remapping, row permutation, and fallback grouping without semantic drift. No production speed claim yet.

### P3 — Bounded census and population accounting

**Depends on:** P2 compact results and P1 oracle.

1. Implement independent S1 top-N with exact omitted levels and row mass; expose it without running S2/S3.
2. Implement deterministic ancestor-closed S2 output, breadth-first budget allocation, per-parent/global selection, and the spec's finite defaults. Tie resolution uses canonical value ordering, never input order.
3. Preserve population counts in post mode while skipping omitted descendants. Count candidate children before exact top-N selection; apply node/level budgets before Python result materialization.
4. Implement global pre masks and per-parent pre selection followed by recomputation on the final common cohort. Output budgets/minimum counts do not redefine that cohort.
5. Keep active depth explicit: dimensions outside `max_depth` do not participate in masks, rankings, or null exclusion.
6. Implement disjoint exclusion counts, expanded-parent conservation, and disjoint frontier accounting. An unexpanded node is not a parent with zero-valued children. Unknown descendant-node totals remain null with `not_computed`.
7. Test originally empty input separately from a nonempty eligible population emptied by pre selection; implement retained-fraction warnings with measured contributions.
8. Instrument candidate/refined row counts and constructed output nodes in benchmarks to demonstrate skipped descendant work without adding timing-sensitive unit assertions.

**Exit:** all pruning combinations preserve exact scope and mass; retained population counts match an unlimited reference; bounds constrain constructed result nodes; no orphan descendants or double-counted omissions.

### P4 — Grain, roles, pairs, and absence

**Depends on:** P1 oracle and P3 scope model.

1. Implement explicit `KeySpec` plus unambiguous single-column shorthand. Support tuple-labelled columns independently of composite keys.
2. Implement FD group/row violation counts, evaluated/singleton/repeated support, undefined empty evaluations, and all compatible determinants. Distinguish observed evidence from semantic entity grain.
3. Compute equivalent and incomparable determinant candidates only on a common row universe; label pair-specific drop-NA comparisons as not comparable unless explicitly recomputed on a shared scope.
4. Keep full input for S3 by default. `explore(..., top_n_applies_to="both")` only accepts a valid pre cohort and records lineage. Reject post+both and no-cohort combinations.
5. Benchmark segmented encoded FD reductions against normalized pandas. Preserve pandas if encoding/sorting does not improve the complete S3 request.
6. Implement role proposals with cardinality/dtype/name-hint reasons and explicit not-evaluated FD evidence. User-selected dimensions override inferred/confirmed role warnings.
7. Implement sparse pair mappings and uncorrected Cramér's V; test against literal/dense oracle tables including empty/constant cases.
8. Reconcile declared domains, calculate exact absent-cell class totals from observed counts and support, and generate bounded deterministic examples without materializing or scanning a huge Cartesian domain. Track omitted requested pairs/contexts separately from cell examples.
9. Include context path and source scope everywhere; pair saturation never asserts freedom from higher-order constraints.

**Exit:** oracle parity, no forced grain assignments, full empty/missing/composite support, bounded large-domain absence, and correct XOR limitation.

### P5 — Public API, renderer, and user documentation

**Depends on:** P3–P4 result semantics.

1. Expose top-level `infer_schema`, `levels`, `census`, `grain`, `explore`, `render_plaintext`, `KeySpec`, and types. Functional and accessor calls delegate to the same implementations.
2. Test calls in fresh processes after ordinary `import bea_tools`, not only after importing a private registration module.
3. Implement rendering from result objects/dicts only. Enforce line/width/node budgets before path expansion and reserve an in-budget truncation marker.
4. Default safe mode escapes non-ASCII labels and terminal controls to printable ASCII; optional display mode uses the `unicode` extra and documented width rules. Quote literal missing-label strings distinctly. Avoid a whole-result `to_dict()`/JSON intermediary just to print a few lines.
5. Add examples for independent top-five counts, bounded nested census, conditional pre filtering, composite keys, explicit schema overrides, reference domains, and strict JSON. Use synthetic inputs only.
6. Explain that exact top-N still scans data, output limits are not evidence of absent combinations, and `undetermined` does not prove finer grain.
7. Update README versions/extras and link the user guide and this delivery plan.

**Exit:** import/accessor parity, deterministic serialized outputs, narrow-width/control-sequence tests, executable documentation examples, and no unconditional plotting/sampling imports in the explorer path.

### P6 — End-to-end benchmarks, CI, and release readiness

**Depends on:** P5 complete API and identical output adapters.

1. Run the benchmark protocol in §7; choose kernels based on correctness plus complete request time/memory, not just aggregation speed.
2. Record selected backend rationale, cold/warm tradeoffs, fixture sizes, output counts, and residual limits in `benchmarks/README.md` or an adjacent benchmark-results document. Keep generated machine JSON reproducible and separate from project/task state.
3. Run core-only and extras-enabled tests, supported-version jobs, lint, source build, and fresh installed-wheel smoke tests.
4. Compare regressions against an equivalent baseline environment when existing tests fail; preserve failure identities, not only counts. A solver/environment problem is not a passing sampler regression.
5. Review the final diff and exported API; preserve unrelated files and release workflow. Implementation commits, pushes, and publishing follow their own authorization.

**Exit:** all acceptance gates pass with recorded evidence, or specific unresolved failures are reported. Passing documentation validation alone does not complete this phase.

## 5. Correctness test inventory

Use descriptive test modules under `bea_tools/testing/explore/`; this is the required coverage map, not a claim that tests have been written.

| Test module | Assertions | Spec reference |
|---|---|---|
| `test_encoding.py` | Typed `True`/`1`/`1.0`/`"1"`; wrapper normalization; one NA; literal `"<NA>"`; exact large ints/floats; signed zero; non-finite tags; temporal/timezone/range behavior; unsupported objects | §4 |
| `test_inputs.py` | Duplicate/unknown/tuple labels, repeated dimensions, invalid and zero budgets, explicit schema warnings, absent required dimensions, composite key syntax | §§4, 5, 7, 9 |
| `test_levels.py` | Independent feature scopes, top-N ties, exact omitted mass, all-null/unused categories, no conjunctive filtering | §5 |
| `test_census.py` | Exact observed prefixes, shallow/single-dimension paths, retained population counts, canonical IDs and parent links | §6 |
| `test_pruning.py` | Ancestor closure, expanded-parent equation, disjoint frontier, global/per-parent pre/post, totals-only requests, intersecting limits, unknown descendant counts, original-empty versus emptied mask | §6 |
| `test_grain.py` | Key/target missingness scopes, group and row rates, singleton support, constant/unique targets, composite keys, no-groups null, equivalent/incomparable determinants, restricted scope lineage | §7 |
| `test_relations.py` | Active-dimension pair selection after depth limits, A/B orientation, directional mapping classes, constant/empty Cramér's V, dense/sparse parity, explicit contexts, XOR limitation | §8 |
| `test_absence.py` | Domain mismatch errors, declared zero-support levels, disjoint class totals, huge sparse domains, no Cartesian allocation/scan, deterministic example/context/pair caps | §8 |
| `test_roles.py` | Reviewable reasons, no false confidence, not-evaluated FD evidence, explicit selection wins | §5 |
| `test_results.py` | Version 0.2, section statuses, compact dictionary references, strict JSON with `allow_nan=False`, no volatile fields by default, explicit undefined reasons | §9 |
| `test_render.py` | Width 1/3/normal, line/node budgets, literal missing labels, safe/native Unicode behavior, multiline/control escaping, no full serialization intermediate | §10 |
| `test_properties.py` | Input immutability, row permutation/category reorder/hash-seed determinism, unpruned duplication scaling/bijective relabeling, exact FD heredity for nonempty evaluated subsets and null/no-evaluated-groups for empty subsets | §12 |
| `test_differential.py` | Independent oracle versus normalized pandas versus each candidate; no shared grouping/encoding logic in expected values | §§3, 12 |
| `test_api.py` and `test_explore_imports.py` | Public import/accessor equivalence, explicit proposal acceptance, combined orchestration, minimal/extras environments | §§9, 11 |

### Fixture design

- **Identifiable planted hierarchy:** at least three entity levels, with repeated groups and deliberate attribute changes across child entities. Ensure the intended coarsest compatible key is identifiable from the sample. Finding-level targets only become undetermined without the finding key when the constructed observations actually vary within coarser groups.
- **Ambiguous hierarchy:** constant attributes, singleton-heavy keys, equivalent keys, reused local IDs, and incomparable keys. Correct output preserves ambiguity.
- **Flat and no-ID tables:** same production code paths; explicit keys can be absent without failure.
- **Noise/violations:** one offending observation and multiple conflicting groups; verify group-weighted and row-weighted rates separately. Do not label all affected rows erroneous.
- **XOR:** pairwise complete support with incomplete three-way occupancy.
- **Large domains:** two high-cardinality features with sparse observed matches and explicit unobserved declared levels. Expected totals are arithmetic; tests must not instantiate the Cartesian domain.
- **Scalar adversaries:** null sentinels, literal display tokens, mixed scalar families, timestamps/timezones, large integers, infinity, empty strings, Unicode combining marks/emoji, ANSI/OSC and bidi control payloads.

Property-based generators should use fixed seeds/reproducible failure examples in CI. Hand-calculated fixtures remain necessary even with differential and randomized tests.

## 6. Packaging and validation commands

The commands below are **planned acceptance commands**, not results from this documentation change. They become runnable after the corresponding files, dependency group, extras, and benchmark CLI have been implemented. Use project-local tooling; do not change the global Python environment.

### Core-only job

```bash
uv sync --group dev
uv run --group dev python -m pytest bea_tools/testing/explore bea_tools/testing/test_explore_imports.py -q
uv run --group dev ruff check bea_tools/_explore bea_tools/testing/explore bea_tools/testing/test_explore_imports.py
uv run --group dev ruff format --check bea_tools/_explore bea_tools/testing/explore bea_tools/testing/test_explore_imports.py
```

Do not collect the existing PuLP-dependent sampler module in the no-extras job. Check that the core-only job really lacks optional dependencies; another job's environment must not leak into it.

### Extras and regression job

```bash
uv sync --group dev --extra sampling --extra plotting --extra unicode
uv run --group dev --extra sampling --extra plotting --extra unicode python -m pytest bea_tools/testing -q
uv build
```

Keep normal pytest discovery to `test_*.py`; the legacy `_pandas.py` script is not silently promoted into a passing baseline. Record optional solver availability and run all three existing sampler tests. If CBC cannot start, report the environment failure and resolve it under the implementation scope instead of marking the tests passed/skipped without disclosure.

### Clean-wheel validation

After `uv build`, create a fresh temporary environment with `uv venv`, install the exact built wheel with `uv pip install --python <fresh-python> <wheel-path>`, and run a smoke script from outside the repository with a clean `PYTHONPATH`. The script must exercise:

- `import bea_tools` without PuLP/matplotlib/wcwidth;
- Series and DataFrame accessor registration;
- independent counts and a small nested/FD result;
- strict JSON serialization;
- the safe bounded renderer;
- an actionable error when an optional feature is requested without its extra.

Use a second isolated environment to install the same wheel with extras and verify sampler/plot exports and native Unicode rendering. Run from outside the source checkout so an editable path cannot hide broken packaging. Record the exact wheel filename/hash and Python/library versions in validation evidence, not in this planning document.

### CI matrix

At minimum test Python 3.11, 3.12, and 3.13 with compatible dependency resolution. Include a Python 3.11 job with explicit exact pandas/NumPy constraints at the declared floors, separately from a current-compatible dependency job. Assert and record installed versions before the floor tests; ordinary `uv sync` against the default lock does not prove that floor versions were exercised. Create the constrained environment without rewriting the normal project lock. Add newer Python versions only after verifying library compatibility; do not equate an open-ended metadata range with tested support. Tests are ordinary CI; hardware-specific performance gates run on the named benchmark machine and are recorded rather than asserted with brittle shared-runner wall-time tests.

## 7. Reproducible benchmark protocol and engine selection

### 7.1 Harness interface

Implement a standalone synthetic-only harness with a documented command such as:

```bash
uv run --group dev python benchmarks/feature_hierarchy_explorer.py --suite acceptance --repeats 7 --output <temporary-results-directory>
```

That CLI is a planned interface, not an existing command. It should expose fixture/backend/workload selection, seed, repetitions, and output location. Store JSON per subprocess plus an aggregate summary. Separate timing runs from allocation-tracing runs because tracing changes time. Assert identical canonical outputs before comparing performance. Never substitute generated benchmark numbers for failed executions.

### 7.2 Fixture matrix

Freeze generator code and independent per-fixture seeds. Record actual observed cardinalities and node counts, not just requested domain sizes.

| Family | Primary sizes | Purpose |
|---|---|---|
| Low-cardinality independent | 150k rows; 3 columns with 2/4/8 levels; 6 columns with 4 levels each | Duplicate-heavy paths and leaf-rollup advantage |
| High-cardinality independent | 150k rows; 6 columns with 100 and separately 1,000 levels each | Near-unique deeper prefixes |
| Mixed-cardinality independent | 150k rows; 17k/50k/2/4/8/10k domain sizes | IDs, small categories, high-cardinality measurements without assuming hierarchy |
| Planted hierarchy | 100k and 150k rows; two candidate keys plus categorical/measurement columns | Representative combined S1/S2/S3 workflow |
| Skewed and null-heavy | 150k rows; same shapes with documented skew/null fractions | Top-N effectiveness and normalization |
| Typed variants | Numeric, string, categorical, nullable, mixed supported scalars | Identity costs and fast-path eligibility |
| Sparse domain | 150k rows; large pair domains; bounded/zero examples | Absence totals without Cartesian work |
| Over-envelope | 12 columns × 150k rows and a bounded larger-row case selected for available memory | Degradation diagnostics, not the six-column latency promise |

Use short fixed-length labels for acceptance fixtures and record their byte lengths. Add long-label stress diagnostics separately; row/column limits alone cannot bound arbitrary string storage or JSON length.

### 7.3 Workload matrix

For the primary fixture families compare:

1. S1 only, default `max_levels=100`, and top five per feature.
2. S2 full requested depth with default safety caps, then explicitly unlimited observed output where feasible.
3. S2 shallow depth and global/per-parent top-N population-preserving expansion.
4. S2 conditional pre mode, including empty and low-retention cases. Report those as semantic outcomes, not artificially fast successful analyses.
5. S3 only: one-key microbenchmark plus the two-key acceptance fixtures, including composite keys.
6. `explore()` through strict JSON with explicit dimensions/keys, default node/level limits, pairs enabled, absence disabled; separately enable bounded absence.
7. Rendering through 200 lines from a large compact result, including a fixture around the historical 446,508-node scale; record the actual node count rather than pretending a regenerated fixture exactly matches it.
8. Cold one-shot calls versus reusable encoding within one combined request. A warm microbenchmark is not a public mutable-dataframe cache requirement.

Separate input construction from the call timer. Include all call-triggered validation/normalization, encoding, aggregation, pruning, ordering, labels, model construction, and requested serialization/rendering. Instrument phases without changing semantics. Compare the exact same requested result and limits across candidate backends.

### 7.4 Memory and timing reporting

- Fresh child process for each memory-sensitive case/backend; record process baseline, post-input memory, absolute peak RSS, and incremental peak attributed to the measured request. Document platform units and limitations of high-water RSS subtraction.
- Use an external sampler or equivalent baseline-aware measurement when input construction's high-water mark would hide later allocation. Report input-only control runs and absolute peaks too; do not silently call a high-water difference an exact allocation measure.
- Report `tracemalloc` peak separately; it is not a replacement for process RSS.
- Warm up, then use seven timed repetitions and report median, minimum, maximum, and spread. Preserve raw records, environment metadata, seed/generator revision, node/dictionary counts, JSON byte length, and rendered line count.
- Use the spec's provisional targets: bounded end-to-end ≤5 s and ≤256 MB incremental peak; two-key S3 ≤250 ms; bounded S1 ≤50 ms; renderer ≤200 lines and ≤250 MB incremental peak; absolute RSS <2 GB for declared in-envelope cases.
- Bind acceptance to the recorded reference machine and fixtures. Over-envelope, unlimited-output, and long-label diagnostics remain visible but are not silently folded into the bounded target.

### 7.5 Backend selection rule

Select encoded observed-prefix refinement for S2 if it passes every semantic test and improves the complete target workload without unacceptable memory regressions. Maintain one production path where possible; keep the normalized pandas reference available to tests. If encoded pandas grouping or leaf rollup wins more practically, record the evidence and choose that implementation without changing the public API. Introduce automatic dispatch only when a measured, stable workload distinction justifies its complexity and every branch passes the same contracts.

Retain the direct S1 and pandas FD paths when they win one-shot workloads. Reuse encoding when the combined request makes it beneficial. The critical decision is representation plus grouping/output strategy, not a blanket pandas-versus-NumPy choice.

### 7.6 Evidence already available, and its limits

The September 15 review ran actual in-memory kernel probes using pandas 3.0.5 / NumPy 2.5.3 / Python 3.13.14 on macOS ARM64. All tested prefix counts matched, including 56 small oracle comparisons; FD probes passed 60 small comparisons. Observed-prefix refinement was competitive at low cardinality and fastest in the tested high/mixed-cardinality kernels. Leaf rollup won the duplicate-heavy cases; repeated `np.unique(axis=0)` was slower. Separate one-shot S1/FD tests favored pandas.

Exact timing tables and fixture descriptions are preserved in spec §13.2. Those timings excluded canonical output construction, serialization, and memory measurement, and the intermediate representations differed. The in-memory scripts are not durable benchmark artifacts. Rebuild reproducible fixtures here and measure the full contract before selecting the production kernel or declaring a performance target passed.

## 8. Review and release-readiness checklist

A final implementation review must verify all of the following against real output:

- Spec revision and JSON schema version match; required API names/accessors exist.
- Every documented population/identity/limit combination has a defined outcome and tests.
- The independent oracle does not share production grouping/encoding defects.
- S1/S2/S3 remain independently callable; explicit dimensions and keys win over inference.
- No full Cartesian pair/domain allocation, sparse-ID `bincount`, unchecked packed overflow, or repeated Python full-path payload in the hot path.
- Bounded results have no orphan nodes, double-counted omission mass, invented exact descendant totals, or concealed conditional counts.
- Empty/constant/singleton-heavy FD and pair results state their limitations.
- Minimal and extras-installed wheels work outside the checkout; existing sampler regressions remain intact.
- Timing includes the requested end product, memory metrics are labelled, and actual benchmark outputs support the selected kernel.
- Renderer control escaping, Unicode model, widths, and line/node bounds pass independently of pandas input.
- Documentation examples execute; user-facing statements describe observed compatibility rather than recovered entity truth or structural impossibility.
- Remaining failures, unsupported scalar ranges, and performance exceptions are explicit. No successful completion is inferred from a plausible tree or a passing aggregation microbenchmark.

## 9. Deferred work and change control

Visualization, entity counting, approximate/statistical inference, out-of-core engines, persistent caches, and constraint-tree export remain outside this increment. Additional UI/renderers consume the compact JSON contract rather than reaching into the dataframe.

Behavioral changes are reconciled in the Cortex specification; implementation choices and measured execution evidence belong alongside code/tests/benchmarks. Keep project lifecycle state in Cortex. A future implementation authorization covers code work at that time; this documentation request alone does not start implementation or authorize repository commits/publishing.
