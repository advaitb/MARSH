# CLAUDE.md — OT-based Microbial Source Tracking (Rust)

Project context for Claude Code. Read this before writing code. Companion design doc:
the "Optimal-Transport Microbial Source Tracking" whitepaper (Google Doc). This file is
the implementation spec; the whitepaper is the rationale.

---

## 1. What we are building

`otst` — a fast Rust CLI + library for **microbial source tracking** that estimates what
fraction of a sink community came from each candidate source, using an **optimal-transport
(tree-Wasserstein / UniFrac) loss** instead of the EM / NMF / GLS approaches used by FEAST,
STENSL, SourceID-NMF, and FastST.

Two things make this different and are the whole point — do not lose them:

1. **The loss is tree-Wasserstein (weighted UniFrac).** Cost of moving mass between taxa =
   their phylogenetic distance. This makes the estimator **drift-robust by construction**:
   a taxon shifting to a close relative is cheap. Never replace this with a plain L2/KL loss.
2. **Uncertainty is a bootstrap wrapper**, made affordable by Rust speed. Uncertainty is a
   first-class output, not an afterthought.

The point estimate reduces to a **convex linear program** (global optimum, no local minima).

---

## 2. Core algorithm (implement exactly this)

### 2.1 Inputs
- A **rooted phylogenetic tree** over the D taxa, with non-negative edge lengths `ℓ_e`
  (E edges). Parsed from Newick.
- **Source profiles** `b_1 … b_K`, each a distribution over the D taxa (from a source count
  table; store both raw counts and normalized profiles — raw depths `m_k` are needed for
  the bootstrap).
- **Sink counts** `y` over the D taxa, depth `n = Σ y`. Sink composition `p = y / n`.

### 2.2 Cumulative edge masses (the key transform)
For each edge `e`, let `L(e)` = set of taxa (leaves) in the subtree below `e`.
- Sink cumulative mass: `s_e = Σ_{j ∈ L(e)} p_j`
- Source-k cumulative mass: `M_ek = Σ_{j ∈ L(e)} b_kj`

Compute these with a single post-order traversal (accumulate leaf masses up to the root).
`M` is an `E × K` matrix; `s` is length `E`. This is O(E · K) and done once per solve.

### 2.3 The point estimate: tree-Wasserstein LAD as an LP
Minimize the weighted L1 (least-absolute-deviations) discrepancy of cumulative masses:

```
minimize_w   Σ_e ℓ_e · | s_e − Σ_k M_ek · w_k |
subject to   w_k ≥ 0,   Σ_k w_k = 1
```

Linearize with per-edge slacks `t_e ≥ 0`:

```
minimize   Σ_e t_e
s.t.       t_e ≥  ℓ_e · ( s_e − Σ_k M_ek w_k )     for all e
           t_e ≥ −ℓ_e · ( s_e − Σ_k M_ek w_k )     for all e
           w_k ≥ 0,   Σ_k w_k = 1
```

Variables: `w` (K) + `t` (E). This is a standard LP. **Global optimum — this is the
headline contrast with EM methods.**

### 2.4 Unknown source
Two versions. **Ship v1 first; it is convex and provable. v2 is the extension.**

- **v1 (fixed background source — convex, default):** add a source `b_0` with a *fixed*
  profile (config: `uniform`, or `metacommunity` = mean of all source profiles). Solve the
  (K+1)-source LP with `Σ w = 1`. `w_0` is then the estimated unknown fraction. Fully convex.
- **v2 (unbalanced / partial OT — extension):** allow `Σ_k w_k ≤ 1`; the deficit
  `w_0 = 1 − Σ_k w_k` is unexplained sink mass, penalized at rate `λ` (config). Keep `b_0`
  fixed (or estimate its shape in an *outer* alternating loop — note that jointly estimating
  `w_0 · b_0` is bilinear and **breaks convexity**, so it must live outside the LP, not inside).
  Do NOT try to make the inner solve estimate the unknown profile; that is a deliberate design
  boundary.

### 2.5 Uncertainty: multinomial bootstrap (modular, parallel)
For `b = 1 … B` (default B = 500), independently:
1. Resample sink: `y* ~ Multinomial(n, p)`, set `p* = y*/n`.
2. **Resample every source too:** `counts_k* ~ Multinomial(m_k, b_k)`, set `b_k* = counts_k*/m_k`.
   **This step is mandatory.** Skipping it ignores source-profile noise and the intervals
   will under-cover when sources are shallow. This is the single most important correctness
   requirement in the whole tool.
3. Recompute cumulative masses and re-solve the LP → `w*(b)`.

Report per source: point estimate, mean, and a 95% interval from the `w*` distribution.
- Near the boundary (`w_k ≈ 0`) percentile intervals under-cover; implement an
  **m-out-of-n** or **BCa** option and report near-zero intervals as one-sided.

The `B` replicates are embarrassingly parallel — run them with `rayon`. This parallelism is
what makes uncertainty affordable and is a core reason we are in Rust.

---

## 3. Crate layout

```
otst/
  Cargo.toml
  src/
    lib.rs            # public API
    tree.rs           # Newick parse, post-order traversal, cumulative-mass computation
    io.rs             # read source/sink count tables (TSV/BIOM-lite), align taxa to tree leaves
    profile.rs        # counts <-> normalized profiles; store depths m_k
    lp.rs             # build + solve the LAD LP (v1 and v2)
    unknown.rs        # background-source construction; unbalanced-OT (v2) penalty wiring
    bootstrap.rs      # multinomial resampling (sink + sources), parallel re-solve, intervals
    estimate.rs       # orchestration: one full run (point estimate + bootstrap)
    cli.rs            # clap CLI
  benches/            # criterion benchmarks vs replicate count / K / n
  tests/              # unit + simulation-based integration tests (see §6)
```

Keep `lp.rs` solver-agnostic behind a small trait so the LP backend can be swapped.

---

## 4. Dependencies (proposed — confirm latest versions at build time)

- **LP solver:** `good_lp` (front-end) with a backend such as `clarabel` or `minilp`, OR
  `clarabel` directly. Requirement: handles equality + inequality constraints, thousands of
  vars, fast repeated solves. Verify the chosen crate compiles and is maintained before
  committing to it.
- `rayon` — data-parallel bootstrap.
- `rand` + `rand_distr` — `Multinomial` sampling for the bootstrap.
- `ndarray` — matrices (the E×K cumulative-mass matrix, etc.).
- A Newick parser (e.g. `newick` / `phylotree`) — or hand-roll a small recursive-descent
  parser in `tree.rs` if the ecosystem crates are thin (Newick is simple; a hand-rolled
  parser avoids a fragile dependency).
- `clap` (derive) — CLI.
- `serde` + `serde_json` — config and structured output.
- `anyhow` / `thiserror` — errors.
- `criterion` (dev) — benchmarks.

> Note: crate availability/versions may have changed. Check crates.io before pinning; do not
> assume an API from memory.

---

## 5. CLI / interface

```
otst estimate \
  --sources sources.tsv \      # rows = taxa, cols = source samples (counts)
  --sink sink.tsv \            # single column of counts, same taxa rows
  --tree tree.nwk \            # Newick over the taxa; OMIT for tree-less data (=> L2 loss)
  --unknown \                  # flag: jointly estimate an unknown source (off by default)
  --sparsity 0.0 \             # reweighted-L1 source selection (only with --unknown)
  --bootstrap 500 \            # B; 0 disables uncertainty (auto-off with --unknown)
  --interval bca \             # {percentile, m-out-of-n, bca}
  --threads 0 \                # 0 = all cores
  --out result.json
```

> **Design converged (post-benchmark cleanup).** The tool exposes exactly two orthogonal axes:
> the **loss** is chosen by whether `--tree` is given (present → tree-Wasserstein / drift-robust;
> absent → plain L2 over taxa, for tree-less OTU tables), and **`--unknown`** is a boolean that
> toggles joint unknown-*profile* estimation (the M5 alternating loop). The earlier fixed-background
> modes (`uniform`, `metacommunity`), the unbalanced-OT deficit (`--lambda`), and the
> `--star-tree`/`--taxonomy`/`--cluster-tree` surrogates were removed: the benchmark showed the
> joint estimator dominates the fixed backgrounds, and a star tree is just an L2 fit dressed up as
> a tree. `--unknown` disables the bootstrap (its intervals are not yet wired).

Output JSON: per-source `{name, proportion, ci_low, ci_high, one_sided}`, the unknown
fraction, the objective value, and run metadata (loss, unknown flag, B, depths, solver, wall-clock).

Library API mirror: `otst::estimate(sources, sink, tree, config) -> SourceTrackingResult`.

---

## 6. Testing & validation (this is how we know it works)

Correctness and the paper's claims are validated by a **simulation harness** — build it early,
it is not optional.

### 6.1 Unit tests
- Tree: cumulative masses sum correctly; leaf/subtree membership; degenerate trees.
- LP: on a trivial 2-source case with a known answer, recovers it; respects the simplex
  (weights ≥ 0, sum = 1 in v1); returns global optimum (compare to brute-force grid on small K).
- Bootstrap: with B=1 and no resampling noise, reproduces the point estimate; multinomial
  resampling has correct marginals.

### 6.2 Simulation harness (`tests/sim` + a small binary)
Generate sinks from sources with **ground-truth weights**:
- Sources: sample from EMP-style profiles (or Dirichlet-drawn profiles for CI speed).
- Build sink = known mixture + a **hidden unknown source** + **controlled drift** (perturb a
  fraction of each source's mass, optionally toward phylogenetic neighbors).
- Sequence-sim: draw sink and source counts at configurable depths `n`, `m`.

### 6.3 The four validation experiments (mirror the whitepaper §7)
1. **Drift sweep (headline):** increase drift; our error stays ~flat while a reimplemented
   FEAST/NMF baseline degrades. Assert our error stays below a threshold up to the predicted
   drift level.
2. **Depth sweep:** zero drift; error decreases at the expected `~1/sqrt(n)` + `1/sqrt(m)` rate.
3. **Coverage:** bootstrap 95% intervals contain the truth ≈95% of the time on held-out sinks.
   **Then generate sinks from a DIFFERENT process than the estimator assumes and report the
   coverage drop honestly.** (Do not tune on the test process — that is the FastST trap.)
4. **Speed:** criterion benchmark of the B-replicate run vs replicate count and K; this is the
   headline systems number.

### 6.4 Baselines
Where feasible, wrap or reimplement FEAST/SourceID-NMF for side-by-side error and speed on the
same simulated data. At minimum, implement a plain-L2 (non-OT) deconvolution baseline in-repo
to isolate the contribution of the tree-Wasserstein loss (ablation).

---

## 7. Conventions & gotchas

- **Never swap the OT loss for L2/KL.** The phylogenetic ground metric is the scientific
  contribution; a Euclidean loss silently discards drift-robustness.
- **Always resample sources in the bootstrap** (§2.5 step 2). This is the top correctness bug
  to avoid.
- **Keep the unknown-profile estimation out of the inner LP** (bilinear → non-convex). v1 fixes
  `b_0`; v2's optional shape estimation is an outer loop only.
- Align taxa between the count tables and the tree leaves explicitly; fail loudly on mismatch,
  and document how unmatched taxa are handled (drop vs attach to root).
- Determinism: thread a seed through the bootstrap so runs are reproducible; record it in output.
- Report the empirical anchor/low-abundance situation and tree source in the output metadata —
  results depend on the tree (whitepaper §8).
- Convexity holds **per fixed ground cost**; do not introduce learned edge costs without
  revisiting the LP guarantees.

---

## 8. Milestones

1. **M1 — skeleton + point estimate (v1).** Newick parse, cumulative masses, LAD LP with fixed
   background source, CLI `estimate`, JSON out. Unit tests pass.
2. **M2 — bootstrap uncertainty.** Sink+source multinomial resampling, parallel re-solve,
   percentile intervals. Coverage test passes on matched-process sims.
3. **M3 — simulation harness + 4 experiments.** Drift/depth/coverage/speed; L2 ablation baseline.
4. **M4 — boundary-corrected intervals (m-out-of-n / BCa)** and honest cross-process coverage.
5. **M5 — v2 unbalanced OT** (penalty λ; optional outer-loop unknown shape) behind a flag.
6. **M6 — polish:** docs, benches vs FEAST/SourceID-NMF, packaging.

Ship M1–M3 before touching v2. Correctness and the drift/coverage story matter more than the
unbalanced-OT extension.

---

## 9. Open items to resolve (do not silently assume)

- Verify no existing "OT / tree-Wasserstein source tracking" tool already exists before making
  novelty claims (search the literature; it moves fast).
- Confirm the chosen LP crate is maintained and fast enough for B×(repeated) solves.
  → **RESOLVED (M1):** `good_lp 1.15` + `microlp 0.4` (pure Rust). Verified by a compiled probe
    to solve the LAD LP exactly. Kept behind the `LpSolver` trait so it can be swapped.
- Decide taxa-vs-tree mismatch policy with the user.
  → **RESOLVED (M1 / M1.5):** taxa are matched to tree leaves by exact name. Unmatched table
    taxa are an **error by default** (`--on-missing error`), or dropped and reported with
    `--on-missing drop`. Tree leaves absent from the tables get count 0. When no phylogeny is
    available or IDs are non-informative, the CLI requires exactly one tree source and offers
    two loud, opt-in fallbacks to a Newick file:
    - `--star-tree`: flat star over observed taxa → the loss reduces to **L1 deconvolution**
      (drift-robustness OFF; warned in output metadata). This is also the non-OT ablation.
    - `--taxonomy <FILE>`: approximate tree from `taxon<TAB>lineage` (`;`-delimited ranks);
      ground metric = number of differing ranks. Partial phylogeny without a tree file.
    The output metadata records `phylogeny` (`newick`/`star`/`taxonomy`) and `warnings`.
