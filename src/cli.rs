//! Command-line interface (clap derive) and structured JSON output.

use crate::bootstrap::{self, BootstrapConfig, IntervalMethod};
use crate::estimate::point_estimate;
use crate::io::{align_sink, align_sources, CountTable, OnMissing};
use crate::lp::GoodLpSolver;
use crate::tree::Tree;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "marsh", version, about = "MARSH: optimal-transport microbial source tracking")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Estimate source proportions for a sink community.
    Estimate(EstimateArgs),
}

#[derive(clap::Args, Debug)]
pub struct EstimateArgs {
    /// Source count table (TSV): rows = taxa, columns = source samples.
    #[arg(long)]
    pub sources: PathBuf,
    /// Sink count table (TSV): a single sample column, same taxa rows.
    #[arg(long)]
    pub sink: PathBuf,
    /// Newick tree over the taxa. When given, MARSH fits under the phylogeny-aware
    /// tree-Wasserstein loss (drift-robust). When OMITTED, there is no phylogeny to exploit and
    /// MARSH fits under a plain L2 loss over taxa — use this for tree-less OTU tables.
    #[arg(long)]
    pub tree: Option<PathBuf>,
    /// Estimate an unknown (unobserved) source jointly with the mixing weights. Off by default
    /// (the named sources must explain all sink mass). Turn this on when a substantial fraction
    /// of the sink may come from sources not in the reference set.
    #[arg(long, default_value_t = false)]
    pub unknown: bool,
    /// Reweighted-ℓ1 source-selection strength (only used with `--unknown`). 0 = off. Larger
    /// values suppress weakly-supported sources — useful with many candidate sources.
    #[arg(long, default_value_t = 0.0)]
    pub sparsity: f64,
    /// Advanced (only with `--unknown`): cap the alternating estimator's outer iterations. Lower
    /// values trade a little accuracy for speed, mainly on the slow `--tree --unknown` path.
    /// Default: 60.
    #[arg(long)]
    pub unmix_max_iters: Option<usize>,
    /// Advanced (only with `--unknown`): convergence tolerance on Σ|Δw| between iterations.
    /// Default: 1e-6. Looser values stop the alternation earlier.
    #[arg(long)]
    pub unmix_tol: Option<f64>,
    /// Print per-iteration convergence of the alternating unknown estimator to stderr.
    #[arg(long, default_value_t = false)]
    pub verbose: bool,
    /// How to handle table taxa that are not tree leaves (only relevant with `--tree`).
    #[arg(long, value_enum, default_value_t = OnMissingArg::Error)]
    pub on_missing: OnMissingArg,
    /// Number of bootstrap replicates B for uncertainty; 0 disables it.
    #[arg(long, default_value_t = 500)]
    pub bootstrap: usize,
    /// Bootstrap interval method.
    #[arg(long, value_enum, default_value_t = IntervalArg::Percentile)]
    pub interval: IntervalArg,
    /// RNG seed for the bootstrap (reproducibility; recorded in output).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Worker threads for the parallel bootstrap; 0 = all logical cores.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// Output JSON path (default: stdout).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum OnMissingArg {
    Error,
    Drop,
}

impl From<OnMissingArg> for OnMissing {
    fn from(a: OnMissingArg) -> Self {
        match a {
            OnMissingArg::Error => OnMissing::Error,
            OnMissingArg::Drop => OnMissing::Drop,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum IntervalArg {
    Percentile,
    MOutOfN,
    Bca,
}

impl From<IntervalArg> for IntervalMethod {
    fn from(a: IntervalArg) -> Self {
        match a {
            IntervalArg::Percentile => IntervalMethod::Percentile,
            IntervalArg::MOutOfN => IntervalMethod::MOutOfN,
            IntervalArg::Bca => IntervalMethod::Bca,
        }
    }
}

// ---- Output schema ----

#[derive(Serialize)]
struct SourceOut {
    name: String,
    proportion: f64,
    /// Mean of the bootstrap distribution (equals `proportion` when bootstrap is disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ci_low: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ci_high: Option<f64>,
    /// True when the interval is one-sided `[0, ci_high]` (near-boundary under-coverage).
    #[serde(skip_serializing_if = "Option::is_none")]
    one_sided: Option<bool>,
}

#[derive(Serialize)]
struct BootstrapMeta {
    replicates: usize,
    replicates_used: usize,
    interval: String,
    seed: u64,
    threads: usize,
}

#[derive(Serialize)]
struct Metadata {
    num_taxa: usize,
    num_sources: usize,
    num_edges: usize,
    sink_depth: f64,
    source_depths: Vec<f64>,
    /// Whether an unknown source was jointly estimated (`--unknown`).
    unknown: bool,
    /// Fitting loss: "tree-wasserstein" (with `--tree`) or "l2" (tree-less).
    loss: String,
    /// Tree source: "newick" or "none (L2)".
    phylogeny: String,
    solver: String,
    dropped_taxa: Vec<String>,
    unobserved_leaves: Vec<String>,
    /// User-facing warnings (e.g. drift-robustness disabled under a star tree).
    warnings: Vec<String>,
    /// Bootstrap run info (absent when bootstrap disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    bootstrap: Option<BootstrapMeta>,
    /// Outer iterations run by the alternating unknown-profile estimator (`--unknown estimated`).
    #[serde(skip_serializing_if = "Option::is_none")]
    unmix_iters: Option<usize>,
    /// Wall-clock seconds for the full estimate (point + bootstrap).
    wall_clock_secs: f64,
}

#[derive(Serialize)]
struct EstimateOut {
    sources: Vec<SourceOut>,
    unknown_fraction: f64,
    objective: f64,
    metadata: Metadata,
}

/// Run the CLI.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Estimate(args) => run_estimate(args),
    }
}

/// The union of taxa observed across the source and sink tables, in first-seen order.
fn observed_taxa(src: &CountTable, sink: &CountTable) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut taxa = Vec::new();
    for t in src.taxa.iter().chain(sink.taxa.iter()) {
        if seen.insert(t.clone()) {
            taxa.push(t.clone());
        }
    }
    taxa
}

/// Resolve the tree. With `--tree` we parse the Newick phylogeny and fit under the
/// tree-Wasserstein loss. Without it, we build a flat star tree purely as an alignment scaffold
/// (a star makes tree-Wasserstein equal to plain L1, and the estimator uses an L2 weight step in
/// this case) — this is the tree-less path for raw OTU tables. Returns the tree, whether a real
/// phylogeny was supplied, a metadata label, and any warnings.
fn resolve_tree(
    args: &EstimateArgs,
    src: &CountTable,
    sink: &CountTable,
) -> Result<(Tree, bool, String, Vec<String>)> {
    let mut warnings = Vec::new();
    if let Some(path) = &args.tree {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading tree {}", path.display()))?;
        let tree = Tree::parse_newick(&text)
            .with_context(|| format!("parsing Newick tree {}", path.display()))?;
        Ok((tree, true, "newick".to_string(), warnings))
    } else {
        let taxa = observed_taxa(src, sink);
        let tree = Tree::build_star(&taxa).context("building star scaffold")?;
        warnings.push(
            "no --tree given: fitting under a plain L2 loss over taxa (no phylogeny to exploit). \
             Provide --tree for the drift-robust tree-Wasserstein loss."
                .to_string(),
        );
        Ok((tree, false, "none (L2)".to_string(), warnings))
    }
}

fn run_estimate(args: EstimateArgs) -> Result<()> {
    // Tables first — the tree-less scaffold needs the observed taxa.
    let src_path = args.sources.display().to_string();
    let sink_path = args.sink.display().to_string();
    let src_table = CountTable::from_tsv_path(&args.sources)
        .with_context(|| format!("reading sources {src_path}"))?;
    let sink_table = CountTable::from_tsv_path(&args.sink)
        .with_context(|| format!("reading sink {sink_path}"))?;

    // Resolve the tree: real phylogeny (--tree) => tree-Wasserstein loss; absent => star
    // scaffold + L2 loss.
    let (tree, has_tree, phylogeny, mut warnings) = resolve_tree(&args, &src_table, &sink_table)?;

    let on_missing: OnMissing = args.on_missing.into();
    let (sources, src_report) = align_sources(&src_table, &tree, on_missing, &src_path)?;
    let (sink, sink_report) = align_sink(&sink_table, &tree, on_missing, &sink_path)?;

    // Configure the rayon thread pool if the user requested a specific count.
    if args.threads != 0 {
        // Best-effort: ignore error if a global pool is already set (e.g. in tests).
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global();
    }

    let start = std::time::Instant::now();

    // The two axes are independent: `--unknown` toggles joint unknown-profile estimation; the
    // presence of `--tree` chooses the loss (tree-Wasserstein vs L2). The unknown estimator does
    // not support the bootstrap (see below).
    let effective_bootstrap = if args.unknown { 0 } else { args.bootstrap };
    let mut unmix_iters: Option<usize> = None;

    let (est, prepared) = if args.unknown {
        // Joint unknown-profile estimation (alternating loop). Use the tree-Wasserstein weight
        // step when a real phylogeny is available (drift-robust); otherwise L2, which fits
        // compositional read data better on tree-less data (best-in-class there).
        let weight_step = if has_tree {
            crate::unmix::WeightStep::TreeWasserstein
        } else {
            crate::unmix::WeightStep::L2
        };
        let cfg = crate::unmix::UnmixConfig {
            sparsity: args.sparsity,
            weight_step,
            verbose: args.verbose,
            max_iters: args.unmix_max_iters.unwrap_or(crate::unmix::UnmixConfig::default().max_iters),
            tol: args.unmix_tol.unwrap_or(crate::unmix::UnmixConfig::default().tol),
            ..Default::default()
        };
        let res = crate::unmix::alternating_estimate(&GoodLpSolver, &tree, &sources, &sink, &cfg)
            .context("running the alternating unknown-profile estimator")?;
        unmix_iters = Some(res.iters);
        // The alternating estimator's uncertainty is not yet wired: re-solving replicates with a
        // frozen profile would be inconsistent with the point estimate (and impractically slow on
        // large tables). Disable the bootstrap and say so, rather than emit mismatched intervals.
        if args.bootstrap > 0 {
            warnings.push(
                "bootstrap intervals are not available with --unknown; reporting the point \
                 estimate only."
                    .to_string(),
            );
        }
        // Freeze the estimated unknown profile so the metadata lines up with the reported estimate.
        let prepared = crate::estimate::Prepared::with_background(
            &tree,
            &sources,
            res.unknown_profile.clone(),
        );
        (res.estimate, prepared)
    } else {
        point_estimate(&GoodLpSolver, &tree, &sources, &sink)
            .context("solving the source-tracking LP")?
    };

    // Point weights in prepared.source_names order (named sources + optional Unknown).
    let point_weights: Vec<f64> = est.sources.iter().map(|s| s.proportion).collect();

    // Bootstrap uncertainty (mandatory source + sink resampling inside).
    let boot_cfg = BootstrapConfig {
        replicates: effective_bootstrap,
        seed: args.seed,
        interval: args.interval.into(),
        alpha: 0.05,
    };
    let boot = bootstrap::run(
        &GoodLpSolver,
        &prepared,
        &sources,
        &sink,
        &point_weights,
        &boot_cfg,
    );
    // name -> interval, for enriching the per-source output
    let interval_by_name: std::collections::HashMap<&str, &bootstrap::SourceInterval> =
        boot.sources.iter().map(|si| (si.name.as_str(), si)).collect();

    // The Unknown source's weight is reported separately as the unknown fraction.
    let unknown_fraction: f64 = est
        .sources
        .iter()
        .filter(|s| s.name == crate::unknown::UNKNOWN_LABEL)
        .map(|s| s.proportion)
        .sum::<f64>();

    let have_bootstrap = effective_bootstrap > 0;
    let sources_out: Vec<SourceOut> = est
        .sources
        .iter()
        .filter(|s| s.name != crate::unknown::UNKNOWN_LABEL)
        .map(|s| {
            let iv = interval_by_name.get(s.name.as_str());
            SourceOut {
                name: s.name.clone(),
                proportion: s.proportion,
                mean: iv.filter(|_| have_bootstrap).map(|i| i.mean),
                ci_low: iv.filter(|_| have_bootstrap).map(|i| i.ci_low),
                ci_high: iv.filter(|_| have_bootstrap).map(|i| i.ci_high),
                one_sided: iv.filter(|_| have_bootstrap).map(|i| i.one_sided),
            }
        })
        .collect();

    let mut dropped = src_report.dropped_taxa.clone();
    dropped.extend(sink_report.dropped_taxa.clone());
    dropped.sort();
    dropped.dedup();

    let mut unobserved = src_report.unobserved_leaves.clone();
    unobserved.extend(sink_report.unobserved_leaves.clone());
    unobserved.sort();
    unobserved.dedup();

    // Surface tree-related warnings on stderr as well as in the JSON metadata.
    for w in &warnings {
        eprintln!("warning: {w}");
    }
    // If taxa were dropped, note it as a warning too.
    if !dropped.is_empty() {
        warnings.push(format!("{} taxon(s) not in tree were dropped", dropped.len()));
    }

    let out = EstimateOut {
        sources: sources_out,
        unknown_fraction,
        objective: est.objective,
        metadata: Metadata {
            num_taxa: tree.num_leaves(),
            num_sources: sources.num_sources(),
            num_edges: tree.num_edges(),
            sink_depth: sink.depth,
            source_depths: sources.profiles.iter().map(|p| p.depth).collect(),
            unknown: args.unknown,
            loss: if has_tree { "tree-wasserstein" } else { "l2" }.to_string(),
            phylogeny,
            solver: "good_lp+microlp".to_string(),
            dropped_taxa: dropped,
            unobserved_leaves: unobserved,
            warnings,
            bootstrap: if have_bootstrap {
                Some(BootstrapMeta {
                    replicates: args.bootstrap,
                    replicates_used: boot.replicates_used,
                    interval: format!("{:?}", boot_cfg.interval),
                    seed: boot.seed,
                    threads: rayon::current_num_threads(),
                })
            } else {
                None
            },
            unmix_iters,
            wall_clock_secs: start.elapsed().as_secs_f64(),
        },
    };

    let json = serde_json::to_string_pretty(&out)?;
    match &args.out {
        Some(path) => {
            std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}
