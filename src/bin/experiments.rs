//! Runnable driver for the whitepaper validation experiments (spec §6.3). Prints tab-separated
//! tables to stdout so results can be piped into a plot or a paper. This is the same logic the
//! `tests/validation.rs` integration tests assert on, but here it sweeps finer and reports
//! numbers rather than asserting thresholds.
//!
//! Usage:
//!   cargo run --release --bin marsh-experiments -- [drift|depth|coverage|all]

use marsh::baseline::l2_deconvolve;
use marsh::bootstrap::{self, BootstrapConfig, IntervalMethod};
use marsh::estimate::Prepared;
use marsh::lp::GoodLpSolver;
use marsh::sim::{generate, l1_error, Scenario, SimConfig};
use marsh::tree::Tree;

/// Solve MARSH against an ARBITRARY tree (not the scenario's own), aligning by leaf name. Used to
/// compare the true tree vs a star tree on the same drifted data.
fn solve_ot_with_tree(sc: &Scenario, tree: &Tree) -> Vec<f64> {
    let prepared = Prepared::new(tree, &sc.sources);
    // sources/sink in the scenario are in the scenario tree's leaf order (T0..T{D-1}); the
    // alternative trees use the same leaf names, and Prepared/cumulative_masses index by the
    // tree's own leaf order, so we must re-map profiles into `tree`'s leaf order.
    let remap = |vec_in_sc_order: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; tree.num_leaves()];
        for (sc_pos, &node) in sc.tree.leaves().iter().enumerate() {
            let name = sc.tree.nodes[node].name.as_ref().unwrap();
            if let Some(tnode) = tree.leaf_by_name(name) {
                // position of tnode in `tree`'s leaf order
                let tpos = tree.leaves().iter().position(|&n| n == tnode).unwrap();
                out[tpos] = vec_in_sc_order[sc_pos];
            }
        }
        out
    };
    let src: Vec<Vec<f64>> = sc
        .sources
        .profiles
        .iter()
        .map(|p| remap(&p.normalized()))
        .collect();
    let sink = remap(&sc.sink.normalized());
    prepared.solve(&GoodLpSolver, &src, &sink).unwrap().weights
}

/// Star tree over a scenario's taxa.
fn star_tree_for(sc: &Scenario) -> Tree {
    let taxa: Vec<String> = sc
        .tree
        .leaves()
        .iter()
        .map(|&n| sc.tree.nodes[n].name.clone().unwrap())
        .collect();
    Tree::build_star(&taxa).unwrap()
}

fn solve_ot(sc: &Scenario) -> Vec<f64> {
    let prepared = Prepared::new(&sc.tree, &sc.sources);
    let src: Vec<Vec<f64>> = sc.sources.profiles.iter().map(|p| p.normalized()).collect();
    prepared
        .solve(&GoodLpSolver, &src, &sc.sink.normalized())
        .unwrap()
        .weights
}

fn solve_l2(sc: &Scenario) -> Vec<f64> {
    let src: Vec<Vec<f64>> = sc.sources.profiles.iter().map(|p| p.normalized()).collect();
    l2_deconvolve(&src, &sc.sink.normalized())
}

fn mean_err(cfg: &SimConfig, n: u64, f: impl Fn(&Scenario) -> Vec<f64>) -> f64 {
    let mut t = 0.0;
    for s in 0..n {
        let sc = generate(cfg, s * 100 + 1);
        t += l1_error(&f(&sc), &sc.true_weights);
    }
    t / n as f64
}

fn drift_sweep() {
    println!("# Experiment 1: drift sweep (OT vs L2 ablation)");
    println!("drift\tOT_l1_err\tL2_l1_err");
    let base = SimConfig {
        num_taxa: 64,
        num_sources: 5,
        dirichlet_alpha: 0.3,
        sink_depth: 20_000,
        source_depth: 20_000,
        ..Default::default()
    };
    let n = 20;
    for i in 0..=8 {
        let drift = i as f64 * 0.05;
        let cfg = SimConfig { drift, ..base.clone() };
        println!(
            "{:.2}\t{:.4}\t{:.4}",
            drift,
            mean_err(&cfg, n, solve_ot),
            mean_err(&cfg, n, solve_l2)
        );
    }
}

fn depth_sweep() {
    println!("# Experiment 2: depth sweep (OT error vs sequencing depth)");
    println!("depth\tOT_l1_err");
    let base = SimConfig {
        num_taxa: 48,
        num_sources: 4,
        dirichlet_alpha: 0.4,
        drift: 0.0,
        ..Default::default()
    };
    let n = 20;
    for &depth in &[250u64, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000] {
        let cfg = SimConfig {
            sink_depth: depth,
            source_depth: depth,
            ..base.clone()
        };
        println!("{}\t{:.4}", depth, mean_err(&cfg, n, solve_ot));
    }
}

fn coverage() {
    println!("# Experiment 3: bootstrap 95% interval coverage");
    println!("scenario_process\tcoverage\tn_intervals");
    // Matched process: sinks generated the same way the estimator assumes (no drift).
    report_coverage("matched(no-drift)", 0.0);
    // Cross-process (spec §6.3): sinks generated with drift the estimator does NOT model.
    // Report the coverage drop honestly.
    report_coverage("cross(drift=0.3)", 0.3);
}

fn report_coverage(label: &str, drift: f64) {
    let cfg = SimConfig {
        num_taxa: 24,
        num_sources: 3,
        dirichlet_alpha: 0.5,
        drift,
        sink_depth: 5_000,
        source_depth: 5_000,
        ..Default::default()
    };
    let n_scenarios = 100u64;
    let (mut covered, mut total) = (0usize, 0usize);
    for seed in 0..n_scenarios {
        let sc = generate(&cfg, seed * 7 + 3);
        let prepared = Prepared::new(&sc.tree, &sc.sources);
        let src: Vec<Vec<f64>> = sc.sources.profiles.iter().map(|p| p.normalized()).collect();
        let point = prepared
            .solve(&GoodLpSolver, &src, &sc.sink.normalized())
            .unwrap()
            .weights;
        let bc = BootstrapConfig {
            replicates: 300,
            seed: 1_000 + seed,
            interval: IntervalMethod::Percentile,
            alpha: 0.05,
        };
        let res = bootstrap::run(&GoodLpSolver, &prepared, &sc.sources, &sc.sink, &point, &bc);
        for (j, si) in res.sources.iter().enumerate() {
            let truth = sc.true_weights[j];
            let lo = if si.one_sided { 0.0 } else { si.ci_low };
            total += 1;
            if truth >= lo - 1e-9 && truth <= si.ci_high + 1e-9 {
                covered += 1;
            }
        }
    }
    println!(
        "{}\t{:.3}\t{}",
        label,
        covered as f64 / total as f64,
        total
    );
}

/// Experiment 5: isolate the phylogeny's contribution. On the SAME drifted data, compare MARSH
/// with the true tree vs a star tree (tree-Wasserstein → L1) vs the L2 baseline. The true-tree
/// column should stay flat under drift while star/L2 degrade.
fn tree_benefit_sweep() {
    println!("# Experiment 5: true tree vs star vs L2 under phylogenetic drift");
    println!("drift\tOT_true\tOT_star\tL2");
    let base = SimConfig {
        num_taxa: 48,
        num_sources: 4,
        dirichlet_alpha: 0.3,
        sink_depth: 20_000,
        source_depth: 20_000,
        ..Default::default()
    };
    let n = 15;
    for i in 0..=6 {
        let drift = i as f64 * 0.05;
        let cfg = SimConfig { drift, ..base.clone() };
        let (mut t_true, mut t_star, mut t_l2) = (0.0, 0.0, 0.0);
        for s in 0..n {
            let sc = generate(&cfg, s * 100 + 1);
            t_true += l1_error(&solve_ot(&sc), &sc.true_weights);
            let st = star_tree_for(&sc);
            t_star += l1_error(&solve_ot_with_tree(&sc, &st), &sc.true_weights);
            t_l2 += l1_error(&solve_l2(&sc), &sc.true_weights);
        }
        let n = n as f64;
        println!("{:.2}\t{:.4}\t{:.4}\t{:.4}", drift, t_true / n, t_star / n, t_l2 / n);
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match which.as_str() {
        "drift" => drift_sweep(),
        "depth" => depth_sweep(),
        "coverage" => coverage(),
        "treebenefit" => tree_benefit_sweep(),
        "all" => {
            drift_sweep();
            println!();
            depth_sweep();
            println!();
            coverage();
            println!();
            tree_benefit_sweep();
        }
        other => {
            eprintln!("unknown experiment {other:?}; use: drift | depth | coverage | treebenefit | all");
            std::process::exit(1);
        }
    }
}
