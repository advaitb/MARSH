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
#[command(group(clap::ArgGroup::new("tree_source").required(true).multiple(false)))]
pub struct EstimateArgs {
    /// Source count table (TSV): rows = taxa, columns = source samples.
    #[arg(long)]
    pub sources: PathBuf,
    /// Sink count table (TSV): a single sample column, same taxa rows.
    #[arg(long)]
    pub sink: PathBuf,
    /// Newick tree over the taxa. Exactly one tree source is required: --tree,
    /// --star-tree, or --taxonomy.
    #[arg(long, group = "tree_source")]
    pub tree: Option<PathBuf>,
    /// Build a flat STAR tree over the observed taxa instead of using a phylogeny. Reduces the
    /// tree-Wasserstein loss to plain L1 deconvolution — phylogenetic drift-robustness is OFF.
    /// Use when no tree is available; taxon IDs need only be unique strings.
    #[arg(long, group = "tree_source")]
    pub star_tree: bool,
    /// Build an approximate tree from a taxonomy file (TSV: taxon<TAB>lineage, lineage is a
    /// ';'-delimited rank string). Gives partial phylogenetic structure without a real tree.
    #[arg(long, group = "tree_source", value_name = "FILE")]
    pub taxonomy: Option<PathBuf>,
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
    /// Tree source: "newick", "star", or "taxonomy".
    phylogeny: String,
    solver: String,
    dropped_taxa: Vec<String>,
    unobserved_leaves: Vec<String>,
    /// User-facing warnings (e.g. drift-robustness disabled under a star tree).
    warnings: Vec<String>,
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

/// Resolve the tree from exactly one of --tree / --star-tree / --taxonomy. Returns the tree, a
/// short phylogeny-type label for the output metadata, and any user-facing warnings.
fn build_tree(
    args: &EstimateArgs,
    src: &CountTable,
    sink: &CountTable,
) -> Result<(Tree, String, Vec<String>)> {
    let mut warnings = Vec::new();

    if let Some(path) = &args.tree {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading tree {}", path.display()))?;
        let tree = Tree::parse_newick(&text)
            .with_context(|| format!("parsing Newick tree {}", path.display()))?;
        Ok((tree, "newick".to_string(), warnings))
    } else if args.star_tree {
        let taxa = observed_taxa(src, sink);
        let tree = Tree::build_star(&taxa).context("building star tree")?;
        warnings.push(
            "star tree in use: the tree-Wasserstein loss reduces to L1 deconvolution; \
             phylogenetic drift-robustness is DISABLED."
                .to_string(),
        );
        Ok((tree, "star".to_string(), warnings))
    } else if let Some(path) = &args.taxonomy {
        let (taxa, lineages) = read_taxonomy(path)?;
        let tree = Tree::build_from_taxonomy(&taxa, &lineages)
            .with_context(|| format!("building tree from taxonomy {}", path.display()))?;
        warnings.push(
            "approximate taxonomy tree in use: ground metric = number of differing ranks, \
             not true branch lengths."
                .to_string(),
        );
        Ok((tree, "taxonomy".to_string(), warnings))
    } else {
        // Unreachable: the clap ArgGroup marks a tree source as required.
        anyhow::bail!("no tree source provided (use --tree, --star-tree, or --taxonomy)")
    }
}

/// Read a taxonomy TSV: `taxon<TAB>lineage` per line, lineage is a ';'-delimited rank string.
fn read_taxonomy(path: &std::path::Path) -> Result<(Vec<String>, Vec<String>)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading taxonomy {}", path.display()))?;
    let mut taxa = Vec::new();
    let mut lineages = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let taxon = parts.next().unwrap_or("").trim().to_string();
        let lineage = parts.next().unwrap_or("").trim().to_string();
        if taxon.is_empty() {
            anyhow::bail!("taxonomy {}: empty taxon id on line {}", path.display(), i + 1);
        }
        taxa.push(taxon);
        lineages.push(lineage);
    }
    if taxa.is_empty() {
        anyhow::bail!("taxonomy {} has no entries", path.display());
    }
    Ok((taxa, lineages))
}

fn run_estimate(args: EstimateArgs) -> Result<()> {
    // Tables first — star/taxonomy tree builders need the observed taxa.
    let src_path = args.sources.display().to_string();
    let sink_path = args.sink.display().to_string();
    let src_table = CountTable::from_tsv_path(&args.sources)
        .with_context(|| format!("reading sources {src_path}"))?;
    let sink_table = CountTable::from_tsv_path(&args.sink)
        .with_context(|| format!("reading sink {sink_path}"))?;

    // Resolve the tree from exactly one source (enforced by the clap ArgGroup).
    let (tree, phylogeny, mut warnings) = build_tree(&args, &src_table, &sink_table)?;

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
            unknown_mode: format!("{mode:?}"),
            phylogeny,
            solver: "good_lp+microlp".to_string(),
            dropped_taxa: dropped,
            unobserved_leaves: unobserved,
            warnings,
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
