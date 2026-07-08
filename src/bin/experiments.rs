//! Runnable driver for the whitepaper validation experiments (spec §6.3). Prints tab-separated
//! tables to stdout so results can be piped into a plot or a paper. This is the same logic the
//! `tests/validation.rs` integration tests assert on, but here it sweeps finer and reports
//! numbers rather than asserting thresholds.
//!
//! Usage:
//!   cargo run --release --bin otst-experiments -- [drift|depth|coverage|all]

use otst::baseline::l2_deconvolve;
use otst::bootstrap::{self, BootstrapConfig, IntervalMethod};
use otst::estimate::Prepared;
use otst::lp::GoodLpSolver;
use otst::sim::{generate, l1_error, Scenario, SimConfig};
use otst::unknown::UnknownMode;

fn solve_ot(sc: &Scenario) -> Vec<f64> {
    let prepared = Prepared::new(&sc.tree, &sc.sources, UnknownMode::None, 0.0);
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
        let prepared = Prepared::new(&sc.tree, &sc.sources, UnknownMode::None, 0.0);
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

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match which.as_str() {
        "drift" => drift_sweep(),
        "depth" => depth_sweep(),
        "coverage" => coverage(),
        "all" => {
            drift_sweep();
            println!();
            depth_sweep();
            println!();
            coverage();
        }
        other => {
            eprintln!("unknown experiment {other:?}; use: drift | depth | coverage | all");
            std::process::exit(1);
        }
    }
}
