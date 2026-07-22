# MARSH — Optimal-Transport Microbial Source Tracking

MARSH estimates the mixing proportions of a microbial **sink** community over a set of candidate
**source** communities. Unlike count-matching methods, it can score the fit under a **phylogeny-aware
tree-Wasserstein loss**, so a sink whose taxa have *drifted* to nearby relatives of the reference
taxa (different 16S region, sequencing technology, strain-level turnover) is still attributed to the
right source. It also jointly estimates an **unknown** (unobserved) source, with a drift-robust,
parameter-free estimator.

Written in Rust; the linear programs are solved with a pure-Rust LP backend (no C/Fortran deps).

## Build

```bash
cargo build --release
# binary: target/release/marsh
```

## Usage

```bash
marsh estimate \
  --sources sources.tsv \      # TSV: rows = taxa, columns = source samples
  --sink    sink.tsv \         # TSV: same taxa rows, a single sink column
  [--tree   tree.nwk] \        # Newick over the taxa -> phylogeny-aware tree-Wasserstein loss
  [--unknown] \                # also estimate an unobserved source
  [--bootstrap 500] \          # bootstrap replicates for uncertainty (0 disables)
  [--out result.json]          # default: stdout
```

- **Without `--tree`**: fits a plain L2 loss over taxa — the right choice for tree-less OTU tables or
  when there is no phylogenetic drift between sources and sink.
- **With `--tree`**: fits the **locality-bounded** tree-Wasserstein loss, which transports drifted
  mass back to the correct source across phylogenetically-local feature mismatch. Transport is
  confined to local clades (edges above the √D-leaf scale are dropped), so the loss stays robust to
  local drift without over-smoothing under global/temporal turnover — the parameter-free scale is
  derived from the tree alone.
- **With `--unknown`**: estimates the fraction of the sink coming from sources not in the reference
  set. On the `--tree --unknown` path this uses a residual-anchored estimator that separates a
  genuine unknown source from drift with no tuned parameter.

Output is JSON: per-source proportions, the unknown fraction (if `--unknown`), the objective, and
bootstrap confidence intervals (if enabled). Run `marsh estimate --help` for the full flag list.

### Input format

`sources.tsv` / `sink.tsv` are tab-separated count (or relative-abundance) tables with a taxon-ID
first column; the sink has one sample column. When `--tree` is given, taxon IDs must match the
Newick leaf labels (`--on-missing drop` tolerates table taxa absent from the tree).

## Method

MARSH casts source tracking as a constrained optimal-transport fit: find source weights (and an
optional unknown profile) whose mixture minimizes the tree-Wasserstein (or L2) distance to the sink,
solved as a linear program. Uncertainty comes from a parallel multinomial bootstrap. See
[`DESIGN.md`](DESIGN.md) for the full specification and rationale.

## Benchmarks & paper experiments

The cross-method benchmark (vs. FEAST, SourceID-NMF, FastST) and all paper experiments — simulated
sweeps and real 16S data (GlobalPatterns), with reproduction scripts and results — live on the
[`benchmarking`](../../tree/benchmarking) branch under `benchmark/` (see `benchmark/README.md`).

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
