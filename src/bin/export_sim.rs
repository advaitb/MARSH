//! Export simulated, TREE-BEARING source-tracking scenarios to disk so the phylogeny-benefit
//! benchmark can run OTST (with the real tree) head-to-head against the tree-blind competitors
//! (FEAST, FastST, SourceID-NMF) on the SAME data.
//!
//! Unlike the FastST/SourceID-NMF datasets, these sims ship a real phylogeny and apply
//! TADA-style phylogenetic drift to the sink — the regime where the tree-Wasserstein loss is
//! supposed to pay off. Each replicate writes four files under `<out_dir>/rep_<i>/`:
//!   sources.tsv  — taxa × K source count columns (OTST/FEAST/etc. input)
//!   sink.tsv     — taxa × 1 sink count column
//!   tree.nwk     — the true Newick tree (fed to OTST via --tree; competitors ignore it)
//!   truth.csv    — header row `unknown,S0,S1,…`; one data row of ground-truth proportions
//!
//! Usage:
//!   export_sim <out_dir> <n_reps> <drift> [num_taxa] [num_sources] [unknown_fraction] [seed_base] [drift_model]
//!
//! Defaults: num_taxa=200, num_sources=6, unknown_fraction=0.0, seed_base=1, drift_model=cherry.
//! `drift_model` is `cherry` (tip-localized sibling-leaf swap; the phylogeny-friendly default) or
//! `tada` (whole-tree Beta-split perturbation). Example:
//!   export_sim benchmark/data/phylo 30 0.3

use otst::sim::{generate, DriftModel, ProfileModel, SimConfig};
use std::fs;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <out_dir> <n_reps> <drift> [num_taxa] [num_sources] [unknown_fraction] [seed_base]",
            args[0]
        );
        std::process::exit(1);
    }
    let out_dir = &args[1];
    let n_reps: usize = args[2].parse().expect("n_reps");
    let drift: f64 = args[3].parse().expect("drift");
    let num_taxa: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(200);
    let num_sources: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(6);
    let unknown_fraction: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let seed_base: u64 = args.get(7).map(|s| s.parse().unwrap()).unwrap_or(1);
    // Default to cherry-swap: tip-localized drift (mass moves to an immediate sibling leaf), the
    // biologically realistic regime where a phylogeny-aware loss helps. TADA-Beta is available
    // for a whole-tree structured perturbation but must be tuned for locality.
    let drift_model = match args.get(8).map(|s| s.as_str()) {
        Some("tada") => DriftModel::TadaBeta,
        _ => DriftModel::CherrySwap,
    };
    // Arg 9: profile model. `tree` = phylogenetically-structured (Dirichlet-tree, the biologically
    // faithful model); anything else = iid Dirichlet (no tree structure, the neutral baseline).
    let profile_model = match args.get(9).map(|s| s.as_str()) {
        Some("tree") => ProfileModel::DirichletTree,
        _ => ProfileModel::IidDirichlet,
    };

    let cfg = SimConfig {
        num_taxa,
        num_sources,
        dirichlet_alpha: 0.3,
        unknown_fraction,
        drift,
        drift_model,
        profile_model,
        sink_depth: 20_000,
        source_depth: 20_000,
    };

    fs::create_dir_all(out_dir).expect("create out_dir");
    for i in 1..=n_reps {
        // Distinct seed per replicate (same scheme as the experiments binary).
        let sc = generate(&cfg, seed_base.wrapping_add(i as u64).wrapping_mul(100).wrapping_add(1));
        let rep_dir = format!("{out_dir}/rep_{i}");
        fs::create_dir_all(&rep_dir).expect("create rep_dir");

        let leaf_names: Vec<String> = sc
            .tree
            .leaves()
            .iter()
            .map(|&n| sc.tree.nodes[n].name.clone().unwrap_or_else(|| format!("T{n}")))
            .collect();

        write_sources(&rep_dir, &leaf_names, &sc.sources);
        write_sink(&rep_dir, &leaf_names, &sc.sink);
        fs::write(format!("{rep_dir}/tree.nwk"), sc.tree.to_newick()).expect("write tree");
        write_truth(&rep_dir, sc.sources.num_sources(), &sc.true_weights, sc.true_unknown);
    }

    // A small manifest for the runner.
    let mut manifest = fs::File::create(format!("{out_dir}/manifest.tsv")).expect("manifest");
    writeln!(manifest, "n_reps\tdrift\tnum_taxa\tnum_sources\tunknown_fraction").unwrap();
    writeln!(
        manifest,
        "{n_reps}\t{drift}\t{num_taxa}\t{num_sources}\t{unknown_fraction}"
    )
    .unwrap();
    eprintln!(
        "wrote {n_reps} replicates to {out_dir} (drift={drift}, taxa={num_taxa}, K={num_sources}, unknown={unknown_fraction}, profiles={profile_model:?})"
    );
}

fn write_sources(dir: &str, taxa: &[String], sources: &otst::profile::SourceSet) {
    let path = Path::new(dir).join("sources.tsv");
    let mut f = fs::File::create(&path).expect("create sources.tsv");
    // header
    write!(f, "taxon").unwrap();
    for name in &sources.names {
        write!(f, "\t{name}").unwrap();
    }
    writeln!(f).unwrap();
    // rows
    for (i, taxon) in taxa.iter().enumerate() {
        write!(f, "{taxon}").unwrap();
        for prof in &sources.profiles {
            write!(f, "\t{}", prof.counts[i] as u64).unwrap();
        }
        writeln!(f).unwrap();
    }
}

fn write_sink(dir: &str, taxa: &[String], sink: &otst::profile::Profile) {
    let path = Path::new(dir).join("sink.tsv");
    let mut f = fs::File::create(&path).expect("create sink.tsv");
    writeln!(f, "taxon\tsink").unwrap();
    for (i, taxon) in taxa.iter().enumerate() {
        writeln!(f, "{taxon}\t{}", sink.counts[i] as u64).unwrap();
    }
}

fn write_truth(dir: &str, k: usize, weights: &[f64], unknown: f64) {
    let path = Path::new(dir).join("truth.csv");
    let mut f = fs::File::create(&path).expect("create truth.csv");
    // header: unknown,S0,S1,...
    write!(f, "unknown").unwrap();
    for j in 0..k {
        write!(f, ",S{j}").unwrap();
    }
    writeln!(f).unwrap();
    // one data row
    write!(f, "{unknown}").unwrap();
    for &w in weights.iter() {
        write!(f, ",{w}").unwrap();
    }
    writeln!(f).unwrap();
}
