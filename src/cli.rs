//! Command-line interface (clap derive) and structured JSON output.

use crate::estimate::point_estimate;
use crate::io::{align_sink, align_sources, CountTable, OnMissing};
use crate::lp::GoodLpSolver;
use crate::tree::Tree;
use crate::unknown::UnknownMode;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "otst", version, about = "Optimal-transport microbial source tracking")]
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
    /// Newick tree over the taxa.
    #[arg(long)]
    pub tree: PathBuf,
    /// Unknown-source model.
    #[arg(long, value_enum, default_value_t = UnknownArg::Metacommunity)]
    pub unknown: UnknownArg,
    /// Penalty λ for unbalanced OT (v2 only).
    #[arg(long, default_value_t = 1.0)]
    pub lambda: f64,
    /// How to handle table taxa that are not tree leaves.
    #[arg(long, value_enum, default_value_t = OnMissingArg::Error)]
    pub on_missing: OnMissingArg,
    /// Output JSON path (default: stdout).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum UnknownArg {
    None,
    Uniform,
    Metacommunity,
    Unbalanced,
}

impl From<UnknownArg> for UnknownMode {
    fn from(a: UnknownArg) -> Self {
        match a {
            UnknownArg::None => UnknownMode::None,
            UnknownArg::Uniform => UnknownMode::Uniform,
            UnknownArg::Metacommunity => UnknownMode::Metacommunity,
            UnknownArg::Unbalanced => UnknownMode::Unbalanced,
        }
    }
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

// ---- Output schema ----

#[derive(Serialize)]
struct SourceOut {
    name: String,
    proportion: f64,
}

#[derive(Serialize)]
struct Metadata {
    num_taxa: usize,
    num_sources: usize,
    num_edges: usize,
    sink_depth: f64,
    source_depths: Vec<f64>,
    unknown_mode: String,
    solver: String,
    dropped_taxa: Vec<String>,
    unobserved_leaves: Vec<String>,
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

fn run_estimate(args: EstimateArgs) -> Result<()> {
    // Tree
    let tree_text = std::fs::read_to_string(&args.tree)
        .with_context(|| format!("reading tree {}", args.tree.display()))?;
    let tree = Tree::parse_newick(&tree_text)
        .with_context(|| format!("parsing Newick tree {}", args.tree.display()))?;

    // Tables
    let src_path = args.sources.display().to_string();
    let sink_path = args.sink.display().to_string();
    let src_table = CountTable::from_tsv_path(&args.sources)
        .with_context(|| format!("reading sources {src_path}"))?;
    let sink_table = CountTable::from_tsv_path(&args.sink)
        .with_context(|| format!("reading sink {sink_path}"))?;

    let on_missing: OnMissing = args.on_missing.into();
    let (sources, src_report) = align_sources(&src_table, &tree, on_missing, &src_path)?;
    let (sink, sink_report) = align_sink(&sink_table, &tree, on_missing, &sink_path)?;

    let mode: UnknownMode = args.unknown.into();

    // Point estimate
    let (est, _prep) = point_estimate(&GoodLpSolver, &tree, &sources, &sink, mode, args.lambda)
        .context("solving the tree-Wasserstein LP")?;

    // Split named sources from the Unknown source for reporting.
    let unknown_fraction: f64 = est
        .sources
        .iter()
        .filter(|s| s.name == crate::unknown::UNKNOWN_LABEL)
        .map(|s| s.proportion)
        .sum::<f64>()
        + est.deficit; // v2 deficit also counts as unknown

    let sources_out: Vec<SourceOut> = est
        .sources
        .iter()
        .filter(|s| s.name != crate::unknown::UNKNOWN_LABEL)
        .map(|s| SourceOut {
            name: s.name.clone(),
            proportion: s.proportion,
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
            unknown_mode: format!("{mode:?}"),
            solver: "good_lp+microlp".to_string(),
            dropped_taxa: dropped,
            unobserved_leaves: unobserved,
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
